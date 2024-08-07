import os
import argparse
import torch
from torch.backends import cudnn
from utils.utils import *
import random
import numpy as np
def setup_seed(seed):
 torch.manual_seed(seed)
 torch.cuda.manual_seed_all(seed)
 np.random.seed(seed)
 random.seed(seed)
 torch.backends.cudnn.deterministic = True

from local_solver import Solver

def str2bool(v):
    return v.lower() in ('true')


def main(config):
    setup_seed(20)
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train_dual()
        # return solver#,best_epoch
    elif config.mode == 'test':
        if config.task=='Reconstruct':
            if len(config.n_group)>1:
                solver.test_dual_reconstruct2_(method='mean')#config.best_epoch)
       
            else:
                solver.test_dual_reconstruct_(method='mean')

        return solver


if __name__ == '__main__':
    

 
    dataset='WADI'#'WADI'
    data_scale='min-max'
    reduce=True
    output_encoding=False#True
    output_attention = True
    R_arch='LSTM'
    assoc_lam=0.0001
    if dataset=='WADI':
        # enc_in=123
        save_location='WADI'
        batch_size=32
        lr=1e-5
        assoc_lam=0.0001
        num_epochs=3
        aratio=1.042
        if reduce:
            enc_in=109
            test_version=3
        else:
            enc_in=123
            test_version=2
    elif dataset=='SWaT':
        batch_size=32
        lr=1e-5
        assoc_lam=0.0001
        num_epochs=3
        save_location='SWaT'#'
        aratio=4.9
        if reduce:
            enc_in=26
            test_version=3
        else:
            enc_in=51
            test_version=2
    # n_group=[6,6]#[6]
    
    # for d_model in [2]:
    #     for n_group in [[6],[6,6]]:#,[6,6,6],[12,6,3],[8],[8,8],[8,8,8],[12,8,4],[24,12,8],[8,4]]:
    
            # print(d_model,n_group)
    n_group = [6]
    nsensor=[enc_in]
    win_size=10
    d_model=512
    sensor_embed='linear'#'conv'
    if len(n_group)>1:
        nsensor.extend(n_group[:-1])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    idx=[torch.arange(i).to(device) for i in nsensor]
    parser = argparse.ArgumentParser()
    # train_method
    parser.add_argument('--train_method', type=str, default='0')#'Together
    parser.add_argument('--nsensor', type=list, default=nsensor)
    parser.add_argument('--n_group', type=list, default=n_group)
    parser.add_argument('--idx', type=list, default=idx)
    parser.add_argument('--e_layers', type=int, default=len(n_group))
    parser.add_argument('--model_name', type=str, default='Dual')
    parser.add_argument('--task', type=str, default='Reconstruct')
    parser.add_argument('--forecast_step', type=float, default=1)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--num_epochs', type=int, default=num_epochs)
    parser.add_argument('--k', type=int, default=1e-3)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--win_size', type=int, default=win_size)
    parser.add_argument('--d_model', type=int, default=d_model)
    parser.add_argument('--enc_in', type=int, default=enc_in)
    parser.add_argument('--c_out', type=int, default=win_size)
    parser.add_argument('--d_ff', type=int, default=2)#512
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--test_version', type=str, default=test_version, choices=[1,2])
    parser.add_argument('--data_path', type=str, default=save_location)#'./dataset/creditcard_ts.csv')
    parser.add_argument('--revin', type=bool, default=False)
    parser.add_argument('--R_arch', type=str, default=R_arch)
    parser.add_argument('--results_save_path', type=str, default='results')
    parser.add_argument('--output_encoding', type=bool, default=output_encoding)
    parser.add_argument('--output_attention', type=bool, default=output_attention)
    parser.add_argument('--sensor_embed', type=str, default=sensor_embed)
    parser.add_argument('--data_scale', type=str, default=data_scale)
    #assoc_lam
    parser.add_argument('--assoc_lam', type=float, default=assoc_lam)
    args = parser.parse_args()
    # print(args)
    Groups='G'
    for i in n_group:
        Groups+=str(i)
    parser.add_argument('--Groups', type=str, default=Groups)
    # model_save_path='TrainedModels/SWaT15'+args.model_name+args.task+args.train_method+'/'
    model_save_path='TrainedModels/thre'+dataset+args.model_name+args.task+args.train_method+Groups+R_arch+'dmodel'+str(d_model)+'/'
    parser.add_argument('--model_save_path', type=str, default=model_save_path)#'checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=aratio)
    parser.add_argument('--device', type=torch.device, default=device)
    parser.add_argument('--reduce', type=float, default=reduce)
    parser.add_argument('--sensor_adj', type=float, default=False)
    config = parser.parse_args()
    parser.add_argument('--args', type=dict, default=config)
    args = vars(config)
    # print('------------ Options -------------')
    # for k, v in sorted(args.items()):
    #     print('%s: %s' % (str(k), str(v)))
    # print('-------------- End ----------------')
    main(config)
    # config.best_epoch=best_epoch
    # print('Test Version', test_version)
    config.mode = 'test'
    main(config)
