import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random


# # EEGNet

# In[25]:
class EM_feature(nn.Module):
    def __init__(self,dropout_rate=0.2):
        super(EM_feature, self).__init__()
        F1 = 32
        Chans = 6
        self.conv1 = nn.Conv2d(1, F1, (Chans,8), stride=(Chans,8))
        self.norm = nn.BatchNorm2d(F1)
        self.norm = nn.LayerNorm(F1*16)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x[:,:,:6,:]))
        x = self.dropout(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    
    

class EEG_feature(nn.Module):
    def __init__(self, input_time=1000, fs=128, ncha=64, filters_per_branch=8,
                 scales_time=(500, 250, 125), dropout_rate=0.25,
                 activation='elu', n_classes=3, learning_rate=0.001):
        super(EEG_feature, self).__init__()

        input_samples = int(input_time * fs / 1000)
        scales_samples = [int(s * fs / 1000) for s in scales_time]  #1*250*64
        scales_samples = [64, 32, 16, 8]

        self.b1_units = nn.ModuleList()
        for i in range(len(scales_samples)):
            unit = nn.Sequential(
                nn.ZeroPad2d(padding=(0,0,scales_samples[i]//2,scales_samples[i]//2-1)),
                nn.Conv2d(1, filters_per_branch, (scales_samples[i], 1)),
                nn.BatchNorm2d(filters_per_branch),
                nn.ELU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(filters_per_branch, filters_per_branch*2, (1, ncha), groups=filters_per_branch),
                nn.Conv2d(filters_per_branch*2, filters_per_branch*2, (1, 1)),
                nn.BatchNorm2d(filters_per_branch*2),
                nn.ELU(inplace=True),
                nn.Dropout(dropout_rate)
            )
            self.b1_units.append(unit)

        self.b2_units = nn.ModuleList()
        for i in range(len(scales_samples)):
            unit = nn.Sequential(
                nn.ZeroPad2d((0,0,int(scales_samples[i] / 4)//2,int(scales_samples[i] / 4)//2-1)),
                nn.Conv2d(len(scales_samples) * filters_per_branch*2, filters_per_branch,
                          (int(scales_samples[i] / 4), 1)),
                nn.BatchNorm2d(filters_per_branch),
                nn.ELU(inplace=True),
                nn.Dropout(dropout_rate)
            )
            self.b2_units.append(unit)


        self.output_layer = nn.Sequential(
            nn.Flatten(),
        )

    def forward(self, x):
        x = x.permute(0,1,3,2)
        
        b1_units_out = []
        for unit in self.b1_units:
            b1_units_out.append(unit(x))
        b1_out = torch.cat(b1_units_out, dim=1)
        b1_out = nn.AvgPool2d((4, 1))(b1_out)

        b2_units_out = []
        for unit in self.b2_units:
            b2_units_out.append(unit(b1_out))
        b2_out = torch.cat(b2_units_out, dim=1)
        b2_out = nn.AvgPool2d((2, 1))(b2_out)

        output = b2_out

        return output, b1_out

def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.kaiming_normal_(layer.weight)




class MTREENet(nn.Module):
    def __init__(self):
        super(MoE_Fusion, self).__init__()
        self.eeg_channels = 64
        self.embedding_eeg_dim = 16 #128
        self.embedding_eye_dim = 16
        self.output_eeg_dim = 32*16 #16*128
        self.output_eye_dim = 32*16 #16*16
        
        self.model_eeg = EEG_feature(dropout_rate=0.2)
        self.model_eye = EM_feature()

        self.fc_b1 = nn.Linear(self.output_eeg_dim*4, 3)
        self.fc_b2 = nn.Linear(self.output_eeg_dim, 3)
        self.fc_b1eye = nn.Linear(self.output_eye_dim, 3)
        
        self.norm_eeg = nn.LayerNorm(self.output_eeg_dim)
        self.norm_eye = nn.LayerNorm(self.output_eye_dim)

        self.multihead_attn = nn.MultiheadAttention(self.embedding_eye_dim, num_heads=2, batch_first=True)
        self.fc_eeg_att = nn.Linear(self.embedding_eeg_dim, self.embedding_eye_dim)
        self.multihead_attn_eeg = nn.MultiheadAttention(self.embedding_eeg_dim, num_heads=2, batch_first=True)
        self.fc_eye_att = nn.Linear(self.embedding_eye_dim, self.embedding_eeg_dim)

        self.fc_eeg = nn.Linear(self.output_eeg_dim, 3)
        self.fc_eye = nn.Linear(self.output_eye_dim, 3)
        self.fc = nn.Linear(self.output_eeg_dim+self.output_eye_dim, 2)

        F1 = 16
        self.gating = nn.Sequential(
            nn.Conv2d(1, F1, (self.eeg_channels+6,8), stride=(self.eeg_channels+6,8)),
            nn.Flatten(),
            nn.Linear(F1*1*16,2),
            nn.Softmax(dim=-1)
        )
        self.gating = nn.Sequential(
            nn.Linear(self.output_eeg_dim+self.output_eye_dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):

        eeg = x[:,:,:self.eeg_channels,:]
        eye = x[:,:,self.eeg_channels:self.eeg_channels+6,:]

        fea_eeg, b1_out = self.model_eeg(eeg)

        fea_eye = self.model_eye(eye)
        fea_eeg = torch.squeeze(fea_eeg)
        fea_eye = torch.squeeze(fea_eye)

        b1_out = nn.Flatten()(b1_out)
        b1_out = self.fc_b1(b1_out)
        b2_out = nn.Flatten()(fea_eeg)
        b2_out = self.fc_b2(b2_out)
        b1_outeye = nn.Flatten()(fea_eye)
        b1_outeye = self.fc_b1eye(b1_outeye)

        fea_eye_att, _ = self.multihead_attn(self.fc_eeg_att(fea_eeg), self.fc_eye_att(fea_eye), self.fc_eye_att(fea_eye))
        fea_eeg_att, _ = self.multihead_attn_eeg(self.fc_eye_att(fea_eye), self.fc_eeg_att(fea_eeg), self.fc_eeg_att(fea_eeg))
        fea_eye = fea_eeg_att + fea_eye
        fea_eeg = fea_eye_att + fea_eeg

        fea_eeg = nn.Flatten()(fea_eeg)
        fea_eye = nn.Flatten()(fea_eye)

        fea_all = torch.concat((fea_eeg,fea_eye),dim=-1)
        output_bin = self.fc(fea_all)

        gates = self.gating(fea_all)
        gates_norm = gates

        output_eeg = self.fc_eeg(fea_eeg*gates_norm[:,0:1])
        output_eye = self.fc_eye(fea_eye*gates_norm[:,1:])
        output = output_eeg + output_eye 

        output_two = F.softmax(output_bin, dim=-1)
        output_two = torch.cat((output_two,output_two[:,1:]),dim=-1)
        output_cls = torch.mul(output,output_two)

        return output_cls, output, output_eeg, output_eye, output_bin, gates_norm