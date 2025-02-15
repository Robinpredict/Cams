"""
Train a single model, 
without freezing gradients and two stage training
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from einops import rearrange
import time
from utils.utils import *
from einops import rearrange
from sklearn.cluster import KMeans
import pandas as pd
from model.AnomalyTransformer import  ADSensor,GlobalSensor,DualADSensor

from model.Transformer import DualGlobalSensor
from data_factory.data_loader import get_loader_segment
from einops import rearrange
import warnings
warnings.filterwarnings("ignore")
def compute_2series_loss(global_series,series,n_heads,n_group):
    series_loss = 0.0
    global_series_loss = 0.0
    
    for u in range(len(global_series)):
        if global_series[u].shape[1]!=n_heads:
            global_=global_series[u].repeat(1,n_heads,1,1)
            global_series[u]=global_
        series_loss += (torch.mean(my_kl_loss(series[u], (
                global_series[u] / torch.unsqueeze(torch.sum(global_series[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                       n_group[u])).detach())) + torch.mean(
            my_kl_loss((global_series[u] / torch.unsqueeze(torch.sum(global_series[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               n_group[u])).detach(),
                       series[u])))
        global_series_loss += (torch.mean(my_kl_loss(
            (global_series[u] / torch.unsqueeze(torch.sum(global_series[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                    n_group[u])),
            series[u].detach())) + torch.mean(
            my_kl_loss(series[u].detach(), (
                    global_series[u] / torch.unsqueeze(torch.sum(global_series[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                           n_group[u])))))
       
    series_loss = series_loss / len(global_series)
    global_series_loss = global_series_loss / len(global_series)
    return series_loss,global_series_loss

def my_kl_loss(p, q,reduce=True):
 
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))

    if reduce:

        return torch.mean(torch.sum(res, dim=-1), dim=1)
    else:
        #B,N
        
        return torch.mean(res,dim=-1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.config=config
        self.__dict__.update(Solver.DEFAULTS, **config)

        
        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               step=self.forecast_step,test_version=self.test_version,
                                               mode='train',task=self.task,
                                               dataset=self.dataset,reduce=self.reduce,
                                               data_scale=self.data_scale)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=8, win_size=self.win_size,
                                              step=self.forecast_step,test_version=self.test_version,
                                              mode='val',task=self.task,
                                              dataset=self.dataset,reduce=self.reduce,data_scale=self.data_scale)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              step=self.forecast_step,test_version=self.test_version,
                                              mode='test',task=self.task,
                                              dataset=self.dataset,reduce=self.reduce,data_scale=self.data_scale)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              step=self.forecast_step,test_version=self.test_version,
                                              mode='thre',task=self.task,
                                              dataset=self.dataset,reduce=self.reduce,data_scale=self.data_scale)

        self.build_model()
        # print('SOLVER',self.device)
        # self.device = device#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        
        if self.model_name=='Local':
            self.model=ADSensor(self.win_size, self.enc_in, self.c_out, d_model=self.d_model, 
                           n_heads=self.n_heads, e_layers=self.e_layers,n_group=self.n_group,
                           nsensor=self.nsensor,forecast_step=self.forecast_step,
                           device=self.device,task=self.task,
                           revin=self.revin,output_encoding=self.output_encoding,R_arch=self.R_arch,
                           sensor_embed=self.sensor_embed,
                           args=self.config)
        elif self.model_name=='Global':
            self.model=GlobalSensor(self.win_size, self.enc_in, self.c_out, d_model=self.d_model, 
                           n_heads=self.n_heads, e_layers=self.e_layers,n_group=self.n_group,
                           nsensor=self.nsensor,forecast_step=self.forecast_step,R_arch=self.R_arch,
                           output_attention=self.output_attention,
                           sensor_embed=self.sensor_embed,
                           device=self.device,task=self.task,args=self.config)
        elif self.model_name=='Dual':
            self.model=DualADSensor(self.win_size, self.enc_in, self.c_out, d_model=self.d_model, 
                           n_heads=self.n_heads, e_layers=self.e_layers,n_group=self.n_group,
                           nsensor=self.nsensor,forecast_step=self.forecast_step,
                           device=self.device,task=self.task,
                           output_attention=self.output_attention,
                           revin=self.revin,output_encoding=self.output_encoding,R_arch=self.R_arch,
                           sensor_embed=self.sensor_embed,
                           config=self.config)
        # elif self.model_name=='Local':
            # self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)
        # if torch.cuda.is_available():
        #     self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, labels) in enumerate(vali_loader):
            # if i%self.batch_size==0:
            input = input_data.float().to(self.device)
            labels=labels.float().to(self.device)
            output, series= self.model(input)
            series_loss = 0.0
            global_series_loss = 0.0
            series_loss,global_series_loss=compute_2series_loss(global_series,series,self.n_heads,self.n_group)
               
            
            fore_loss1 = self.criterion(output, labels)
            fore_loss2 = self.criterion(global_output, labels)
            if self.train_method=='Together':
                loss1=fore_loss1+fore_loss2+self.k*global_series_loss+self.k*series_loss
                loss2=loss1
            else:
                loss1 = fore_loss1# - self.k * series_loss
                loss2 = fore_loss2 + self.k * global_series_loss
            loss_1.append(loss1)
            loss_2.append(loss2)

        return np.average(loss_1), np.average(loss_2)
    def compute_2series_loss(self,global_series,series):
        series_loss = 0.0
        global_series_loss = 0.0
        
        for u in range(len(global_series)):
            # print(u)
            for h in range(self.n_heads):
                
                #torch.Size([32, 1, 20, 10]) torch.Size([32, 20, 10]) torch.Size([1, 32, 20, 10])
                # print(series[u].shape,global_series[u].shape,torch.unsqueeze(torch.sum(global_series[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                       # self.n_group[u]).shape)
                series_loss += (torch.mean(my_kl_loss(series[u][:,h,:,:], (
                        global_series[u] / torch.unsqueeze(torch.sum(global_series[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.n_group[u])).detach())) + torch.mean(
                    my_kl_loss((global_series[u] / torch.unsqueeze(torch.sum(global_series[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.n_group[u])).detach(),
                               series[u][:,h,:,:])))
                global_series_loss += (torch.mean(my_kl_loss(
                    (global_series[u] / torch.unsqueeze(torch.sum(global_series[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.n_group[u])),
                    series[u][:,h,:,:].detach())) + torch.mean(
                    my_kl_loss(series[u][:,h,:,:].detach(), (
                            global_series[u] / torch.unsqueeze(torch.sum(global_series[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.n_group[u])))))
            # print(u,series[u][:,h,:,:].shape,global_series[u].shape)
        series_loss = series_loss / len(global_series)
        global_series_loss = global_series_loss / len(global_series)
        return series_loss,global_series_loss
    def train_epoch(self,time_now,epoch_time,epoch,train_steps):
        iter_count=0
        loss1_list=[]
        # epoch_time = time.time()
        for i, (input_data, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            iter_count += 1
            input = input_data.float().to(self.device)
            labels=labels.float().to(self.device)
            # print("1:{}".format(torch.cuda.memory_allocated(device=self.device)))
            if self.output_encoding:
                
                y_pred ,_ = self.model(input)
                del _
            else:
                y_pred  = self.model(input)
            
            loss = self.criterion(y_pred, labels)
            
            loss.backward()
           
            loss1_list.append(loss.item())
           
            self.optimizer.step()
            if (i + 1) % 1000 == 0:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            # print("3:{}".format(torch.cuda.memory_allocated(device=self.device)))
        train_loss = np.average(loss1_list)
        print(epoch,train_loss)
    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        # path = self.model_save_path
        if self.reduce:
            path = self.model_save_path+'reduce'
        else:
            path=self.model_save_path
        print('Train',path)
        if not os.path.exists(path):
            os.makedirs(path)
        # early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)
        # print(train_steps)
        for epoch in range(self.num_epochs):
            # iter_count = 0
            # loss1_list = []
            epoch_time = time.time()
            self.model.train()
            # print("1:{}".format(torch.cuda.memory_allocated(0)))
            self.train_epoch(time_now,epoch_time ,epoch,train_steps)
            

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # train_loss = np.average(loss1_list)
         
            # print(path)
            torch.save(self.model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
            torch.cuda.empty_cache()
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
    def train_dual_epoch(self,time_now,epoch_time,epoch,train_steps):
        iter_count=0
        loss1_list=[]
        global_list=[]
        local_list=[]
        # epoch_time = time.time()
        for i, (input_data, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            iter_count += 1
            input = input_data.float().to(self.device)
            labels=labels.float().to(self.device)
            # print("1:{}".format(torch.cuda.memory_allocated(device=self.device)))
            if self.output_attention:
                
                local_enc_out,local_assoc,global_enc_out,global_assoc= self.model(input)
            else:
                local_enc_out,global_enc_out = self.model(input)
            # print("F:{}".format(torch.cuda.memory_allocated(device=self.device)))
            # calculate Association discrepancy
            
            # series_loss,global_series_loss=compute_2series_loss(global_series,series,self.n_heads,self.n_group)
            
            # print(y_pred.shape,labels.shape)
            local_loss = self.criterion(local_enc_out, labels)
            global_loss = self.criterion(global_enc_out, labels)
            #selection descrepancy
            #batch,nhead,window,window
            #local_assoc: B,H,N,Ngroup
            #global_assoc: B,N,Ngroup
            local_assoc,global_assoc=local_assoc[-1][:,0,:,:],global_assoc[-1]
            
            local_assoc_loss = (torch.mean(my_kl_loss(local_assoc, 
                                                (global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat( 1, 1,
                                                                                           self.n_group[-1])).detach())) + torch.mean(
                my_kl_loss(
                    (global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, 
                                                                                            self.n_group[-1])).detach(),
                    local_assoc)))
            global_assoc_loss = (torch.mean(
                my_kl_loss((global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, 
                                                                                                   self.n_group[-1])),
                           local_assoc.detach())) + torch.mean(
                my_kl_loss(local_assoc.detach(),
                           (global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, 
                                                                                                   self.n_group[-1])))))
            assoc_loss = local_assoc_loss + global_assoc_loss
            loss = local_loss+global_loss+self.assoc_lam*assoc_loss
            # if self.dataset == 'WADI':
            #     print(local_loss,global_loss,assoc_loss)
            loss.backward()
           
            loss1_list.append(loss.item())
            global_list.append(global_loss.item())
            local_list.append(local_loss.item())
           
            self.optimizer.step()
            if (i + 1) % 5000 == 0:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                # print(local_loss.item(),global_loss.item(),assoc_loss.item())
                iter_count = 0
                time_now = time.time()
        train_loss = np.average(loss1_list)
        global_loss = np.average(global_list)
        local_loss = np.average(local_list)
        print(epoch,train_loss,global_loss,local_loss)
    def train_dual(self):

        print("======================TRAIN DUAL MODE======================")

        time_now = time.time()
        # path = self.model_save_path
        if self.reduce:
            path = self.model_save_path+'reduce'
        else:
            path=self.model_save_path
        print('Train',path)
        if not os.path.exists(path):
            os.makedirs(path)
        # early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)
        # print(train_steps)
        for epoch in range(self.num_epochs):
            # iter_count = 0
            # loss1_list = []
            epoch_time = time.time()
            self.model.train()
            # print("1:{}".format(torch.cuda.memory_allocated(0)))
            self.train_dual_epoch(time_now,epoch_time ,epoch,train_steps)
            

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
           
            torch.save(self.model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
            # torch.cuda.empty_cache()
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
    

    
    def test_reconstruct_(self,method='median'):
        if self.reduce:
            if 'reduce' not in self.model_save_path:
                self.model_save_path = self.model_save_path+'reduce/'
        else:
            self.model_save_path=self.model_save_path
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction='none')

        attens_energy = []
        for i, (input_data, y) in enumerate(self.train_loader):
            if i % self.win_size==0:
                input = input_data.float().to(self.device)
                y=y.float().to(self.device)
                if self.output_encoding:
                    y_pred, series= self.model(input)
                else:
                    y_pred= self.model(input)
                #B,Nsensor,step
                
                if method=='max':
                    loss = torch.max(criterion(y_pred, y),dim=1)[0]
                elif method=='median':
                    loss = torch.median(criterion(y_pred, y),dim=1)[0]
                elif method=='mean':
                    loss = torch.mean(criterion(y_pred, y),dim=1)
                #loss:B window
                # print('loss',loss.shape)
                cri = 1 * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
        #28, 51, 1) torch.Size([28, 51, 1])
        
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, y,labels) in enumerate(self.thre_loader):
        # for i, (input_data, y) in enumerate(self.test_loader):
            # if i % self.forecast_step==0:
            input = input_data.float().to(self.device)
            y = y.float().to(self.device)
            if self.output_encoding:
                y_pred, series= self.model(input)
            else:
                y_pred= self.model(input)
            if method=='max':
                loss = torch.max(criterion(y_pred, y),dim=1)[0]
            elif method=='median':
                loss = torch.median(criterion(y_pred, y),dim=1)[0]
            elif method=='mean':
                loss = torch.mean(criterion(y_pred, y),dim=1)
            
            # metric=loss
            # print(metric.shape,loss.shape)
            cri = 1 * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        
    
        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data,ys, labels) in enumerate(self.thre_loader):
            # if i % self.forecast_step==0:
            input = input_data.float().to(self.device)
            ys = ys.float().to(self.device)
            if self.output_encoding:
                y_pred, series= self.model(input)
            else:
                y_pred= self.model(input)
            # y_pred=rearrange(y_pred, 'b n s -> b (n s)')
            # labels=rearrange(labels, 'b n s -> b (n s)')
            # loss = torch.mean(criterion(labels, output), dim=-1)
            #get max
            # loss = torch.max(criterion(y_pred, ys),dim=1)[0]
            if method=='max':
                loss = torch.max(criterion(y_pred, ys),dim=1)[0]
            elif method=='median':
                loss = torch.median(criterion(y_pred, ys),dim=1)[0]
            elif method=='mean':
                loss = torch.mean(criterion(y_pred, ys),dim=1)
            
           
            cri = 1 * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # print(attens_energy.shape)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        # print(test_ener)
        with open(self.results_save_path+'/'+self.dataset+self.model_name+str(self.test_version)+'Task'+self.task+'.npy', 'wb') as f:
            np.save(f, test_energy)
        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
     
        # print(
        #     "Before PA Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        #         accuracy, precision,
        #         recall, f_score))
        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        #a state flag to track whether the current sequence of data points is considered an anomaly
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        # print(
        #     "After PA Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        #         accuracy, precision,
        #         recall, f_score))
 
        return accuracy, precision, recall, f_score
    
    def test_dual_reconstruct_(self,method='median'):
        if self.reduce:
            if 'reduce' not in self.model_save_path:
                self.model_save_path = self.model_save_path+'reduce/'
        else:
            self.model_save_path=self.model_save_path
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()


        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction='none')

        attens_energy = []
        for i, (input_data, y) in enumerate(self.train_loader):
            if i % self.win_size==0:
                input = input_data.float().to(self.device)
                y=y.float().to(self.device)
                if self.output_attention:
                    
                    local_enc_out,local_assoc,global_enc_out,global_assoc= self.model(input)
                else:
                    local_enc_out,global_enc_out = self.model(input)
                # calculate Association discrepancy

                local_loss = criterion(local_enc_out, y)
                global_loss = criterion(global_enc_out, y)
                #selection descrepancy
                #batch,nhead,window,window
                #local_assoc: B,H,N,Ngroup
                #global_assoc: B,N,Ngroup
                local_assoc,global_assoc=local_assoc[-1][:,0,:,:],global_assoc[-1]
                # print('assoc shape',local_assoc.size(),global_assoc.size())
                #B,N,1
                
                local_assoc_loss = my_kl_loss(local_assoc, (global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat( 1, 1,self.n_group[-1])).detach(),reduce=False) +\
                                   my_kl_loss((global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, self.n_group[-1])).detach(),
                        local_assoc,reduce=False)
                global_assoc_loss = my_kl_loss((global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, self.n_group[-1])),
                               local_assoc.detach(),reduce=False) + my_kl_loss(local_assoc.detach(),(global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, 
                                                                                                       self.n_group[-1])),reduce=False)
                assoc_loss = local_assoc_loss + global_assoc_loss
                
                loss = local_loss+global_loss+self.assoc_lam*assoc_loss.unsqueeze(2).repeat(1,1,self.win_size)
                #B,Nsensor,step
               
                if method=='max':
                    loss = torch.max(loss,dim=1)[0]
                elif method=='median':
                    loss = torch.median(loss,dim=1)[0]
                elif method=='mean':
                    loss = torch.mean(loss,dim=1)
                #loss:B window
                # print('loss',loss.shape)
                cri = 1 * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
        #28, 51, 1) torch.Size([28, 51, 1])
        
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, y,labels) in enumerate(self.thre_loader):
        # for i, (input_data, y) in enumerate(self.test_loader):
            # if i % self.forecast_step==0:
            input = input_data.float().to(self.device)
            y = y.float().to(self.device)
            if self.output_attention:
                
                local_enc_out,local_assoc,global_enc_out,global_assoc= self.model(input)
            else:
                local_enc_out,global_enc_out = self.model(input)
            # calculate Association discrepancy

            local_loss = criterion(local_enc_out, y)
            global_loss = criterion(global_enc_out, y)
            
            local_assoc,global_assoc=local_assoc[-1][:,0,:,:],global_assoc[-1]
           
            local_assoc_loss = my_kl_loss(local_assoc, (global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat( 1, 1,
                                                                                           self.n_group[-1])).detach(),reduce=False) +my_kl_loss(
                    (global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, self.n_group[-1])).detach(),
                    local_assoc,reduce=False)
            global_assoc_loss = my_kl_loss((global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, 
                                                                                                   self.n_group[-1])),
                           local_assoc.detach(),reduce=False) + my_kl_loss(local_assoc.detach(),(global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, 
                                                                                                   self.n_group[-1])),reduce=False)
            assoc_loss = local_assoc_loss + global_assoc_loss
            
            loss = local_loss+global_loss+self.assoc_lam*assoc_loss.unsqueeze(2).repeat(1,1,self.win_size)
            #B,Nsensor,step
           
            if method=='max':
                loss = torch.max(loss,dim=1)[0]
            elif method=='median':
                loss = torch.median(loss,dim=1)[0]
            elif method=='mean':
                loss = torch.mean(loss,dim=1)
            
            # metric=loss
            # print(metric.shape,loss.shape)
            cri = 1 * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        f1s=[]
        ratios=[]
        # for anomaly_ratio in np.arange(30,60,2):
            # self.anomaly_ratio=anomaly_ratio/10
            # print(self.anomaly_ratio)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        
    
        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data,ys, labels) in enumerate(self.thre_loader):
            # if i % self.forecast_step==0:
            input = input_data.float().to(self.device)
            ys = ys.float().to(self.device)
            if self.output_attention:
                
                local_enc_out,local_assoc,global_enc_out,global_assoc= self.model(input)
            else:
                local_enc_out,global_enc_out = self.model(input)
            # calculate Association discrepancy

            local_loss = criterion(local_enc_out, ys)
            global_loss = criterion(global_enc_out, ys)
            #selection descrepancy
            #batch,nhead,window,window
            #local_assoc: B,H,N,Ngroup
            #global_assoc: B,N,Ngroup
            local_assoc,global_assoc=local_assoc[-1][:,0,:,:],global_assoc[-1]
         
            local_assoc_loss = my_kl_loss(local_assoc, 
                                                (global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat( 1, 1,
                                                                                           self.n_group[-1])).detach(),reduce=False) +my_kl_loss(
                    (global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, self.n_group[-1])).detach(),
                    local_assoc,reduce=False)
            global_assoc_loss = my_kl_loss((global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, 
                                                                                                   self.n_group[-1])),
                           local_assoc.detach(),reduce=False) + my_kl_loss(local_assoc.detach(),(global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, 
                                                                                                   self.n_group[-1])),reduce=False)
            assoc_loss = local_assoc_loss + global_assoc_loss
            
            loss = local_loss+global_loss+self.assoc_lam*assoc_loss.unsqueeze(2).repeat(1,1,self.win_size)
            #B,Nsensor,step
           
            if method=='max':
                loss = torch.max(loss,dim=1)[0]
            elif method=='median':
                loss = torch.median(loss,dim=1)[0]
            elif method=='mean':
                loss = torch.mean(loss,dim=1)
            
            # metric=global_series_loss
            # print('Test',metric,loss)
            cri = 1 * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
        
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # print(attens_energy.shape)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        # print(test_ener)
        with open(self.results_save_path+'/'+self.dataset+self.model_name+str(self.test_version)+'Task'+self.task+'.npy', 'wb') as f:
            np.save(f, test_energy)
        
        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)
        with open(self.results_save_path+'/'+self.dataset+self.model_name+str(self.test_version)+'AF_Binary_pre'+'.npy', 'wb') as f:
            np.save(f, pred)
        with open(self.results_save_path+'/Label'+self.dataset+self.model_name+str(self.test_version)+'.npy', 'wb') as f:
            np.save(f, gt)
        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        bf_pa_perf=np.array([accuracy,precision,recall,f_score])
        bf_pa_df=pd.DataFrame(bf_pa_perf.reshape(1,-1),columns=['Acc','Pre','Rec','F1'])
        # print('DF',bf_pa_df)
        # bf_pa_df.to_csv(self.results_save_path+'/BF_PA'+self.dataset+self.model_name+str(self.test_version)+'.csv')
        # f1s.append(f_score)
        # ratios.append(anomaly_ratio)
        # print(
        #     "Before PA Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        #         accuracy, precision,
        #         recall, f_score))
        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        #a state flag to track whether the current sequence of data points is considered an anomaly
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)
        with open(self.results_save_path+'/'+self.dataset+self.model_name+str(self.test_version)+'BF_Binary_pre'+'.npy', 'wb') as f:
            np.save(f, pred)
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        # print(
        #     "After PA Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        #         accuracy, precision,
        #         recall, f_score))
        af_pa_perf=np.array([accuracy,precision,recall,f_score])
        af_pa_df=pd.DataFrame(af_pa_perf.reshape(1,-1),columns=['Acc','Pre','Rec','F1'])
        # af_pa_df.to_csv(self.results_save_path+'/AF_PA'+self.dataset+self.model_name+str(self.test_version)+'.csv')
        # print(max(f1s))
        # print(ratios[f1s.index(max(f1s))])
        return accuracy, precision, recall, f_score
    
    def test_dual_reconstruct2_(self,method='median'):
        if self.reduce:
            if 'reduce' not in self.model_save_path:
                self.model_save_path = self.model_save_path+'reduce/'
        else:
            self.model_save_path=self.model_save_path
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction='none')

        attens_energy = []
        for i, (input_data, y) in enumerate(self.train_loader):
            if i % self.win_size==0:
                input = input_data.float().to(self.device)
                y=y.float().to(self.device)
                if self.output_attention:
                    
                    local_enc_out,local_assoc,global_enc_out,global_assoc= self.model(input)
                else:
                    local_enc_out,global_enc_out = self.model(input)
                # calculate Association discrepancy

                local_loss = criterion(local_enc_out, y)
                global_loss = criterion(global_enc_out, y)
                #selection descrepancy
                #batch,nhead,window,window
                #local_assoc: B,H,N,Ngroup
                #global_assoc: B,N,Ngroup
                local_assoc,global_assoc=local_assoc[-1][:,0,:,:],global_assoc[-1]
                # print('assoc shape',local_assoc.size(),global_assoc.size())
                #B,N,1
                
                local_assoc_loss = my_kl_loss(local_assoc, (global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat( 1, 1,self.n_group[-1])).detach(),reduce=False) +\
                                   my_kl_loss((global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, self.n_group[-1])).detach(),
                        local_assoc,reduce=False)
                global_assoc_loss = my_kl_loss((global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, self.n_group[-1])),
                               local_assoc.detach(),reduce=False) + my_kl_loss(local_assoc.detach(),(global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, 
                                                                                                       self.n_group[-1])),reduce=False)
                assoc_loss = torch.mean(local_assoc_loss + global_assoc_loss,dim=1).unsqueeze(1)
                #B,Nsensor
               
                loss = local_loss+global_loss+self.assoc_lam*assoc_loss.unsqueeze(2).repeat(1,self.enc_in,self.win_size)
                #B,Nsensor,step
               
                if method=='max':
                    loss = torch.max(loss,dim=1)[0]
                elif method=='median':
                    loss = torch.median(loss,dim=1)[0]
                elif method=='mean':
                    loss = torch.mean(loss,dim=1)
                #loss:B window
                # print('loss',loss.shape)
                cri = 1 * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
        #28, 51, 1) torch.Size([28, 51, 1])
        
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, y,labels) in enumerate(self.thre_loader):
        # for i, (input_data, y) in enumerate(self.test_loader):
            # if i % self.forecast_step==0:
            input = input_data.float().to(self.device)
            y = y.float().to(self.device)
            if self.output_attention:
                
                local_enc_out,local_assoc,global_enc_out,global_assoc= self.model(input)
            else:
                local_enc_out,global_enc_out = self.model(input)
            # calculate Association discrepancy

            local_loss = criterion(local_enc_out, y)
            global_loss = criterion(global_enc_out, y)
            #selection descrepancy
            #batch,nhead,window,window
            #local_assoc: B,H,N,Ngroup
            #global_assoc: B,N,Ngroup
            local_assoc,global_assoc=local_assoc[-1][:,0,:,:],global_assoc[-1]
           
            local_assoc_loss = my_kl_loss(local_assoc, (global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat( 1, 1,
                                                                                           self.n_group[-1])).detach(),reduce=False) +my_kl_loss(
                    (global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, self.n_group[-1])).detach(),
                    local_assoc,reduce=False)
            global_assoc_loss = my_kl_loss((global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, 
                                                                                                   self.n_group[-1])),
                           local_assoc.detach(),reduce=False) + my_kl_loss(local_assoc.detach(),(global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, 
                                                                                                   self.n_group[-1])),reduce=False)
            
            assoc_loss = torch.mean(local_assoc_loss + global_assoc_loss,dim=1).unsqueeze(1)
            loss = local_loss+global_loss+self.assoc_lam*assoc_loss.unsqueeze(2).repeat(1,self.enc_in,self.win_size)
            #B,Nsensor,step
           
            if method=='max':
                loss = torch.max(loss,dim=1)[0]
            elif method=='median':
                loss = torch.median(loss,dim=1)[0]
            elif method=='mean':
                loss = torch.mean(loss,dim=1)
            
            # metric=loss
            # print(metric.shape,loss.shape)
            cri = 1 * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        
    
        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data,ys, labels) in enumerate(self.thre_loader):
            # if i % self.forecast_step==0:
            input = input_data.float().to(self.device)
            ys = ys.float().to(self.device)
            if self.output_attention:
                
                local_enc_out,local_assoc,global_enc_out,global_assoc= self.model(input)
            else:
                local_enc_out,global_enc_out = self.model(input)
            # calculate Association discrepancy

            local_loss = criterion(local_enc_out, ys)
            global_loss = criterion(global_enc_out, ys)
            #selection descrepancy
            #batch,nhead,window,window
            #local_assoc: B,H,N,Ngroup
            #global_assoc: B,N,Ngroup
            local_assoc,global_assoc=local_assoc[-1][:,0,:,:],global_assoc[-1]
            # print('assoc shape',local_assoc.size(),global_assoc.size())
            #B,N,1
            # print(torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).size())
            local_assoc_loss = my_kl_loss(local_assoc, 
                                                (global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat( 1, 1,
                                                                                           self.n_group[-1])).detach(),reduce=False) +my_kl_loss(
                    (global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, self.n_group[-1])).detach(),
                    local_assoc,reduce=False)
            global_assoc_loss = my_kl_loss((global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, 
                                                                                                   self.n_group[-1])),
                           local_assoc.detach(),reduce=False) + my_kl_loss(local_assoc.detach(),(global_assoc / torch.unsqueeze(torch.sum(global_assoc, dim=-1), dim=-1).repeat(1, 1, 
                                                                                                   self.n_group[-1])),reduce=False)
            assoc_loss = torch.mean(local_assoc_loss + global_assoc_loss,dim=1).unsqueeze(1)
            loss = local_loss+global_loss+self.assoc_lam*assoc_loss.unsqueeze(2).repeat(1,self.enc_in,self.win_size)
            #B,Nsensor,step
           
            if method=='max':
                loss = torch.max(loss,dim=1)[0]
            elif method=='median':
                loss = torch.median(loss,dim=1)[0]
            elif method=='mean':
                loss = torch.mean(loss,dim=1)
            
            # metric=global_series_loss
            # print('Test',metric,loss)
            cri = 1 * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
        
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # print(attens_energy.shape)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        # print(test_ener)
        with open(self.results_save_path+'/'+self.dataset+self.model_name+str(self.test_version)+'Task'+self.task+'.npy', 'wb') as f:
            np.save(f, test_energy)
        with open(self.results_save_path+'/Label'+self.dataset+self.model_name+str(self.test_version)+'.npy', 'wb') as f:
            np.save(f, test_labels)
        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        bf_pa_perf=np.array([accuracy,precision,recall,f_score])
        bf_pa_df=pd.DataFrame(bf_pa_perf.reshape(1,-1),columns=['Acc','Pre','Rec','F1'])
        # print('DF',bf_pa_df)
        # bf_pa_df.to_csv(self.results_save_path+'/BF_PA'+self.dataset+self.model_name+str(self.test_version)+'.csv')
        
        # print(
        #     "Before PA Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        #         accuracy, precision,
        #         recall, f_score))
        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        #a state flag to track whether the current sequence of data points is considered an anomaly
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        # print(
        #     "After PA Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        #         accuracy, precision,
        #         recall, f_score))
        af_pa_perf=np.array([accuracy,precision,recall,f_score])
        af_pa_df=pd.DataFrame(af_pa_perf.reshape(1,-1),columns=['Acc','Pre','Rec','F1'])
        # af_pa_df.to_csv(self.results_save_path+'/AF_PA'+self.dataset+self.model_name+str(self.test_version)+'.csv')
  
        return accuracy, precision, recall, f_score
    
    def test_reconstruct2_(self,method='median',lam=0.001,approach='value'):
        #approach: value kmenas
        # lam=0.001
        if self.reduce:
            self.model_save_path = self.model_save_path+'reduce'
        else:
            self.model_save_path=self.model_save_path
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()


        print("======================TEST MODE======================")
        
        if approach=='kmeans':
            encs=[]
            for i, (input_data, y) in enumerate(self.train_loader):
                if i % self.win_size==0:
                # if i%1==0:
                    input = input_data.float().to(self.device)
                    y=y.float().to(self.device)
                    y_pred, series= self.model(input)
                    encs.append(series)
            encs=torch.cat(encs,0)
            encs=rearrange(encs,'B G L-> (B L) G')
            # print('encs',encs.size())
            method=self.get_normal(encs,n_clusters=20)

        criterion = nn.MSELoss(reduction='none')

        attens_energy = []
        # print('test_reconstruct2')
        encs=[]
        for i, (input_data, y) in enumerate(self.train_loader):
            if i % self.win_size==0:
            # if i%1==0:
                input = input_data.float().to(self.device)
                y=y.float().to(self.device)
                if self.output_encoding:
                    y_pred, series= self.model(input)
                else:
                    y_pred= self.model(input)
                # encs.append(series)
               
                if method=='max':
                    loss = torch.max(criterion(y_pred, y),dim=1)[0]
                    loss2=self.get_loss2(series,approach,method)
                    loss+=lam*loss2#torch.max(abs(series),dim=1)[0]
                elif method=='median':
                    loss = torch.median(criterion(y_pred, y),dim=1)[0]
                    loss2=self.get_loss2(series,approach,method)
                    loss+=lam*loss2
                    # loss+=lam*torch.median(abs(series),dim=1)[0]
                elif method=='mean':
                    loss = torch.mean(criterion(y_pred, y),dim=1)
                    loss2=self.get_loss2(series,approach,method)
                    loss+=lam*loss2
                else:
                    loss =0* torch.mean(criterion(y_pred, y),dim=1)
                    loss2=self.get_loss2(series,approach,method)
                    loss+=lam*loss2
                    # loss+=lam*torch.mean(abs(series),dim=1)
                #loss:B window
                # print('loss',loss.shape)
                cri = 1 * loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
                
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, y,labels) in enumerate(self.thre_loader):
        # for i, (input_data, y) in enumerate(self.test_loader):
            # if i % self.forecast_step==0:
            input = input_data.float().to(self.device)
            y = y.float().to(self.device)
            if self.output_encoding:
                y_pred, series= self.model(input)
            else:
                y_pred= self.model(input)
            # y_pred=rearrange(y_pred, 'b n s -> b (n s)')
            # labels=rearrange(labels, 'b n s -> b (n s)')
            # loss = torch.mean(criterion(labels, output), dim=-1)
            # loss = torch.max(criterion(y_pred, y),dim=1)[0]
            if method=='max':
                loss = torch.max(criterion(y_pred, y),dim=1)[0]
                loss2=self.get_loss2(series,approach,method)
                loss+=lam*loss2
                # loss+=lam*torch.max(abs(series),dim=1)[0]
            elif method=='median':
                loss = torch.median(criterion(y_pred, y),dim=1)[0]
                loss2=self.get_loss2(series,approach,method)
                loss+=lam*loss2
                # loss+=lam*torch.median(abs(series),dim=1)[0]
            elif method=='mean':
                loss = torch.mean(criterion(y_pred, y),dim=1)
                loss2=self.get_loss2(series,approach,method)
                loss+=lam*loss2
                # loss+=lam*torch.mean(abs(series),dim=1)
            else:
                loss = 0*torch.mean(criterion(y_pred, y),dim=1)
                loss2=self.get_loss2(series,approach,method)
                loss+=lam*loss2
            
            # metric=loss
            # print(metric.shape,loss.shape)
            cri = 1 * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
    
        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data,ys, labels) in enumerate(self.thre_loader):
            # if i % self.forecast_step==0:
            input = input_data.float().to(self.device)
            ys = ys.float().to(self.device)
            if self.output_encoding:
                y_pred, series= self.model(input)
            else:
                y_pred= self.model(input)
            #get max
            # loss = torch.max(criterion(y_pred, ys),dim=1)[0]
            if method=='max':
                loss = torch.max(criterion(y_pred, ys),dim=1)[0]
                loss2=self.get_loss2(series,approach,method)
                loss+=lam*loss2
                # loss+=lam*torch.max(abs(series),dim=1)[0]
            elif method=='median':
                loss = torch.median(criterion(y_pred, ys),dim=1)[0]
                loss2=self.get_loss2(series,approach,method)
                loss+=lam*loss2
                # loss+=lam*torch.median(abs(series),dim=1)[0]
            elif method=='mean':
                loss = torch.mean(criterion(y_pred, ys),dim=1)
                loss2=self.get_loss2(series,approach,method)
                loss+=lam*loss2
                # loss+=lam*torch.mean(abs(series),dim=1)
            else:
                loss = 0*torch.mean(criterion(y_pred, y),dim=1)
                loss2=self.get_loss2(series,approach,method)
                loss+=lam*loss2
            
            # metric=global_series_loss
            # print('Test',metric,loss)
            cri = 1 * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # print(attens_energy.shape)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        # print(test_ener)
        with open(self.results_save_path+'/'+self.dataset+self.model_name+str(self.test_version)+'Task'+self.task+'.npy', 'wb') as f:
            np.save(f, test_energy)
        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        # print("pred:   ", pred.shape)
        # print("gt:     ", gt.shape)
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        # f1s.append(f_score)
        # ratios.append(anomaly_ratio)
        bf_pa_perf=np.array([accuracy,precision,recall,f_score])
        bf_pa_df=pd.DataFrame(bf_pa_perf.reshape(1,-1),columns=['Acc','Pre','Rec','F1'])
        # print('DF',bf_pa_df)
        bf_pa_df.to_csv(self.results_save_path+'/BF_PA'+self.dataset+self.model_name+str(self.test_version)+'.csv')
        # print(
        #     "Before PA Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        #         accuracy, precision,
                # recall, f_score))
        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        #a state flag to track whether the current sequence of data points is considered an anomaly
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        # print("pred: ", pred.shape)
        # print("gt:   ", gt.shape)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        # print(
        #     "After PA Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        #         accuracy, precision,
        #         recall, f_score))
        af_pa_perf=np.array([accuracy,precision,recall,f_score])
        af_pa_df=pd.DataFrame(af_pa_perf.reshape(1,-1),columns=['Acc','Pre','Rec','F1'])
        # af_pa_df.to_csv(self.results_save_path+'/AF_PA'+self.dataset+self.model_name+str(self.test_version)+'.csv')

        return accuracy, precision, recall, f_score
