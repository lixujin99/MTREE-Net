#!/usr/bin/env python
# encoding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import argparse
from model import MTREENet
import warnings
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import scipy.io as sio
import numpy as np
import gc
import copy
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from scipy import signal
import scipy.io as sio
import time
from load_data import load_data_sd


def contribution(labels, probs, probs_eeg, probs_eye):
    label_onehot = F.one_hot(labels, num_classes=3)
    
    probs_eeg = F.softmax(probs_eeg, dim=-1)
    probs_eye = F.softmax(probs_eye, dim=-1)
    probs = probs_eeg + probs_eye

    probs = torch.sum(probs*label_onehot, dim=-1)
    probs_eeg = torch.sum(probs_eeg*label_onehot, dim=-1)
    probs_eye = torch.sum(probs_eye*label_onehot, dim=-1)

    weights_eeg = probs_eeg / probs
    weights_eye = probs_eye / probs

    weights = torch.concat((weights_eeg.unsqueeze(1),weights_eye.unsqueeze(1)), dim=-1)

    return weights, weights_eeg, weights_eye


def self_distillation(tri1, bin1):
    tri = F.softmax(tri1, dim=-1)
    bin = F.softmax(bin1, dim=-1)

    tri_output = torch.ones_like(bin, device=bin.device)
    tri_output[:,0] = tri[:,0]
    tri_output[:,1] = tri[:,1] + tri[:,2]

    loss = 0.5*F.kl_div(torch.log(tri_output), bin, reduction='batchmean') + 0.5*F.kl_div(torch.log(bin), tri_output, reduction='batchmean')
    #loss = F.cosine_embedding_loss(tri_output, bin, torch.ones(bin.shape[0], device=bin.device))

    return loss




parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=120, help="number of epochs of training") 
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")  
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")  
parser.add_argument("--b1", type=float, default=0.9, help="adam: learning rate")  
parser.add_argument("--b2", type=float, default=0.999, help="adam: learning rate")  
parser.add_argument("--wd", type=float, default=0.0, help="weight_dec")  
parser.add_argument('--sp', default=1, type=int)
parser.add_argument('--chans', default=64, type=int)
parser.add_argument('--samples', default=128, type=int)
parser.add_argument('--cuda', default=2, type=int)
parser.add_argument('--gamma', default=5, type=int)
parser.add_argument('--resample', default=128, type=int)
parser.add_argument('--model', default='MTREENet', type=str)
parser.add_argument('--patience', default=100, type=int)
opt = parser.parse_args()
print(opt)
#
binary_cls = False
Down_sample = True

if binary_cls:
    cls_nb = 2
else:
    cls_nb = 3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_folder = './save/%s/' %(opt.model)
os.makedirs(model_save_folder, exist_ok=True)
 
Fold = 5
num_block = 5

all_sub = ['S%s'%(str(i+1)) for i in range(43)]

Task_all = ['A', 'B', 'C']

perf1 = np.zeros((len(Task_all), len(all_sub), num_block, Fold))
ba1 = np.zeros((len(Task_all), len(all_sub), num_block, Fold))
conf1 = np.zeros((len(Task_all), len(all_sub), num_block, Fold, cls_nb, cls_nb))


for itask, Task in enumerate(Task_all):
    all_blk = range(1,6)

    for isub, sub in enumerate(all_sub):
        print(sub)
        data, label, id_blk = load_data_sd(Task, sub, all_blk, opt.resample)

        label_bin = np.where(label==0, 0, 1)
        label_all = np.concatenate((np.expand_dims(label,axis=1),np.expand_dims(label_bin,axis=1)),axis=-1)

        data_t_train_mean = data.mean(-1)[:, :, np.newaxis]
        data_t_train_std = data.std(-1)[:, :, np.newaxis] + 1e-6
        data_all = (data - data_t_train_mean) / data_t_train_std
        data_all = data_all[:, np.newaxis, :, :]

        print(data_all.shape)
        print(label_all.shape)

        for id_block in range(num_block):

            index_train = np.where(id_blk==id_block, 0, 1)
            train_data_ori = data_all[index_train==1, :]
            train_label_ori = label_all[index_train==1]
            test_data = data_all[index_train==0, :]
            test_label = label_all[index_train==0]
            test_data = torch.FloatTensor(test_data).to(device)
            
            sample_rate = 1

            Data0 = train_data_ori[train_label_ori[:,0]==0,:]
            Data1 = train_data_ori[train_label_ori[:,0]==1,:]
            Data2 = train_data_ori[train_label_ori[:,0]==2,:]
            Label0 = train_label_ori[train_label_ori[:,0]==0,:]
            Label1 = train_label_ori[train_label_ori[:,0]==1,:]
            Label2 = train_label_ori[train_label_ori[:,0]==2,:]

            num_sample = min(Data0.shape[0],Data1.shape[0],Data2.shape[0])
            rd_index0 = np.random.permutation(Data0.shape[0])
            rd_index1 = np.random.permutation(Data1.shape[0])
            rd_index2 = np.random.permutation(Data2.shape[0])

            data_0_downsampled = Data0[rd_index0[:num_sample*sample_rate],:]
            label_0_downsampled = Label0[rd_index0[:num_sample*sample_rate],:]
            data_1_downsampled = Data1[rd_index1[:num_sample],:]
            label_1_downsampled = Label1[rd_index1[:num_sample],:]
            data_2_downsampled = Data2[rd_index2[:num_sample],:]
            label_2_downsampled = Label2[rd_index2[:num_sample],:]
            
            train_data_ori = np.concatenate((data_0_downsampled,data_1_downsampled,data_2_downsampled),axis=0)
            train_label_ori = np.concatenate((label_0_downsampled,label_1_downsampled,label_2_downsampled),axis=0)

            print(train_data_ori.shape)
            print(train_label_ori.shape)

            kf = StratifiedKFold(n_splits=Fold, shuffle=True)

            for fd, (i_train, i_val) in enumerate(kf.split(train_data_ori, train_label_ori[:,0])):
                train_data = train_data_ori[i_train, :]
                train_label = train_label_ori[i_train]
                val_data = train_data_ori[i_val, :]
                val_label = train_label_ori[i_val]

                model_save = model_save_folder + sub + '_' + str(id_block) + '_' + str(fd) + '_%s_sd.pkl'%(Task)      
                
                class_weights_tri = class_weight.compute_class_weight('balanced', classes=np.unique(train_label[:,0]), y=train_label[:,0])
                class_weights_bin = class_weight.compute_class_weight('balanced', classes=np.unique(train_label[:,1]), y=train_label[:,1])
                class_weights_prb_tri = torch.tensor(class_weights_tri, dtype=torch.float32).to(device)
                class_weights_prb_bin = torch.tensor(class_weights_bin, dtype=torch.float32).to(device)
                criterion = nn.CrossEntropyLoss(weight=class_weights_prb_tri).to(device)
                criterion_bin = nn.CrossEntropyLoss(weight=class_weights_prb_bin).to(device)

                criterion_moe = nn.L1Loss().to(device)

                model_multi = MTREENet()

                model = model_multi.to(device)

                train_data = torch.FloatTensor(train_data)
                train_label = torch.FloatTensor(train_label)
                val_data = torch.FloatTensor(val_data) 
                val_label = torch.FloatTensor(val_label)
                EEG_dataset1 = torch.utils.data.TensorDataset(train_data,train_label)
                EEG_dataset2 = torch.utils.data.TensorDataset(val_data,val_label)
                trainloader = torch.utils.data.DataLoader(EEG_dataset1, batch_size=opt.batch_size, shuffle=True)
                valloader = torch.utils.data.DataLoader(EEG_dataset2, batch_size=opt.batch_size, shuffle=False)

                print("\r", "Task: %s | test subject: [%d/%d//%s] | fold: [%d/%d]" % (Task, isub + 1, len(all_sub), sub, fd+1, Fold))

                
                ############################################################# Train #############################################################
                step = opt.lr
                val_max = 10000
                stepp_new = 0

                optimizer = optim.Adam(model.parameters(), lr=step, weight_decay=opt.wd, betas=(opt.b1, opt.b2))
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
                
                for i in range(opt.epochs):      
                    t = time.time()

                    Label = None
                    Label_bin = None
                    pre = None
                    pre_prob = None
                    pre_prob_eeg = None
                    pre_prob_eye = None
                    pre_prob_bin = None
                    weights_coef = None
                    train_l_sum = 0
                    for ii1, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        inputs = inputs.to(device)
                        outputs_label, outputs_label_sum, outputs_label_eeg, outputs_label_eye, outputs_label_bin, gates = model(inputs)
                        weights_moe, weights_eeg, weights_eye = contribution(labels[:,0].long(), outputs_label_sum, outputs_label_eeg, outputs_label_eye)

                        loss = criterion(outputs_label, labels[:,0].long()) + 0.2*(criterion(outputs_label_eeg, labels[:,0].long()) + criterion(outputs_label_eye, labels[:,0].long()))
                        loss_bin = criterion_bin(outputs_label_bin, labels[:,1].long()) + self_distillation(outputs_label_sum, outputs_label_bin)
                        loss_reweighting = criterion_moe(weights_moe,gates)

                        loss = loss + loss_reweighting + loss_bin
                        
                        loss.backward()
                        optimizer.step()

                        Label = labels[:,0] if Label is None else torch.cat((Label,labels[:,0]))
                        Label_bin = labels[:,1] if Label_bin is None else torch.cat((Label_bin,labels[:,1]))
                        pre_prob = F.softmax(outputs_label,dim=-1) if pre_prob is None else torch.cat((pre_prob,F.softmax(outputs_label,dim=-1)))
                        pre_prob_eeg = F.softmax(outputs_label_eeg,dim=-1) if pre_prob_eeg is None else torch.cat((pre_prob_eeg,F.softmax(outputs_label_eeg,dim=-1)))
                        pre_prob_eye = F.softmax(outputs_label_eye,dim=-1) if pre_prob_eye is None else torch.cat((pre_prob_eye,F.softmax(outputs_label_eye,dim=-1)))
                        pre_prob_bin = F.softmax(outputs_label_bin,dim=-1) if pre_prob_bin is None else torch.cat((pre_prob_bin,F.softmax(outputs_label_bin,dim=-1)))
                        weights_coef =  weights_moe if weights_coef is None else torch.cat((weights_coef,weights_moe))
                        train_l_sum += loss.cpu().item()

                    train_l_sum = train_l_sum / (ii1+1)
                    pre_prob = pre_prob.cpu().detach().numpy()
                    pre_prob_eeg = pre_prob_eeg.cpu().detach().numpy()
                    pre_prob_eye = pre_prob_eye.cpu().detach().numpy()
                    pre_prob_bin = pre_prob_bin.cpu().detach().numpy()
                    BN = balanced_accuracy_score(Label.cpu().detach().numpy(), pre_prob.argmax(-1))
                    BN_eeg = balanced_accuracy_score(Label.cpu().detach().numpy(), pre_prob_eeg.argmax(-1))
                    BN_eye = balanced_accuracy_score(Label.cpu().detach().numpy(), pre_prob_eye.argmax(-1))
                    BN_bin = balanced_accuracy_score(Label_bin.cpu().detach().numpy(), pre_prob_bin.argmax(-1))

                    
                    ############################################################# Validation #############################################################
                    Label = None
                    pre = None
                    pre_prob = None
                    pre_prob_eeg = None
                    pre_prob_eye = None
                    val_l_sum = 0

                    with torch.no_grad():
                        for ii2, data in enumerate(valloader, 0):
                            val_inputs, val_labels = data
                            val_labels = val_labels.to(device)
                            val_inputs = val_inputs.to(device)
                            val_output, val_output_sum, val_output_eeg, val_output_eye, val_output_bin, _ = model(val_inputs)
                            loss_val = criterion(val_output, val_labels[:,0].long())

                            Label = val_labels[:,0] if Label is None else torch.cat((Label,val_labels[:,0]))
                            pre_prob = F.softmax(val_output,dim=-1) if pre_prob is None else torch.cat((pre_prob,F.softmax(val_output,dim=-1)))
                            pre_prob_eeg = F.softmax(val_output_eeg,dim=-1) if pre_prob_eeg is None else torch.cat((pre_prob_eeg,F.softmax(val_output_eeg,dim=-1)))
                            pre_prob_eye = F.softmax(val_output_eye,dim=-1) if pre_prob_eye is None else torch.cat((pre_prob_eye,F.softmax(val_output_eye,dim=-1)))
                            val_l_sum += loss.cpu().item()

                    val_l_sum = val_l_sum / (ii1+1)
                    pre_prob = pre_prob.cpu().detach().numpy()
                    pre_prob_eeg = pre_prob_eeg.cpu().detach().numpy()
                    pre_prob_eye = pre_prob_eye.cpu().detach().numpy()
                    val_BN = balanced_accuracy_score(Label.cpu().detach().numpy(), pre_prob.argmax(-1))
                    val_BN_eeg = balanced_accuracy_score(Label.cpu().detach().numpy(), pre_prob_eeg.argmax(-1))
                    val_BN_eye = balanced_accuracy_score(Label.cpu().detach().numpy(), pre_prob_eye.argmax(-1))

                    scheduler.step(val_BN)

                    if ((i+1)%10==0):
                        print('Epoch: {:04d}'.format(i+1),
                            'loss_train: {:.4f}'.format(train_l_sum),
                                "BN= {:.4f}".format(BN),
                                "BN_eeg= {:.4f}".format(BN_eeg),
                                "BN_eye= {:.4f}".format(BN_eye),
                                "BN_bin= {:.4f}".format(BN_bin),
                                'loss_val: {:.4f}'.format(val_l_sum),
                                "val_BN= {:.4f}".format(val_BN),
                                "val_BN_eeg= {:.4f}".format(val_BN_eeg),
                                "val_BN_eye= {:.4f}".format(val_BN_eye),
                                "time: {:.4f}s".format(time.time() - t))
                    
                    if (val_l_sum<val_max):
                        val_max = val_l_sum
                        stepp_new = 0
                        torch.save(model.state_dict(), model_save)
                    else:
                        stepp_new = stepp_new + 1
                    
                    if stepp_new>opt.patience:
                        break

                print('Finished Training')
                model.load_state_dict(torch.load(model_save, map_location=device))

                ############################################################# Test #############################################################
                with torch.no_grad():
                    probs, probs_sum, probs_eeg, probs_eye, probs_bin, weights = model(test_data)
                    weights_moe, weights_eeg, weights_eye = contribution(torch.FloatTensor(test_label[:,0]).long().to(device), probs_sum, probs_eeg, probs_eye)
                probs = probs.cpu().detach().numpy()
                probs_eeg = probs_eeg.cpu().detach().numpy()
                probs_eye = probs_eye.cpu().detach().numpy()
                probs_bin = probs_bin.cpu().detach().numpy()
                clf_perf2 = accuracy_score(test_label[:,0], probs.argmax(-1))
                clf_ba = balanced_accuracy_score(test_label[:,0], probs.argmax(-1))
                clf_ba_eeg = balanced_accuracy_score(test_label[:,0], probs_eeg.argmax(-1))
                clf_ba_eye = balanced_accuracy_score(test_label[:,0], probs_eye.argmax(-1))
                clf_ba_bin = balanced_accuracy_score(test_label[:,1], probs_bin.argmax(-1))
                clf_conf = confusion_matrix(test_label[:,0], probs.argmax(-1))

                perf1[itask, isub, id_block, fd] = clf_perf2
                ba1[itask, isub, id_block, fd] = clf_ba
                conf1[itask, isub, id_block, fd, :] = clf_conf


                print("\r", "Task: %s | test subject: [%d/%d//%s] | fold: [%d/%d]" % (Task, isub + 1, len(all_sub), sub, fd+1, Fold))
                print("\r", "Test  : ACC:%.4f" % (clf_perf2))
                print("\r", "Test : BA:%.4f" % (clf_ba), "BA_eeg:%.4f" % (clf_ba_eeg), "BA_eye:%.4f" % (clf_ba_eye), "BA_bin:%.4f" % (clf_ba_bin))
                print("\r", clf_conf)


            mean_fold = np.mean(ba1,axis=-1)
            mean_sub = np.mean(mean_fold,axis=-1)
            mean_task = np.mean(mean_sub,axis=-1)
            print('Block')
            print(mean_fold)
            print('Subject')
            print(mean_sub)
            print('Task')
            print(mean_task)


result_name = './%s_tri.mat'%(opt.model)
sio.savemat(result_name, {'perf123': perf1, 'ba123': ba1, 'conf123': conf1})
