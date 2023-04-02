import time
import argparse
import sys
import os
import os.path as osp

import numpy as np
import torch
import pandas as pd

from training.gnn_y import train_gnn_y
from training.gnn_mdi import train_gnn_mdi

from training.knn import train_knn
from training.mlp import train_mlp
from training.DynEdgConv import train_dec

from uji.uji_subparser import add_uji_subparser
from utils.utils import auto_select_gpu

epochs = 20000
unlab_train = 0.4
lr = 0.0015
log_dir = f"feat_missing_{unlab_train}_lr_{lr}_b1f1"

# Finding Label using feature imputation for missing data points.

parser = argparse.ArgumentParser()
parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
parser.add_argument('--post_hiddens', type=str, default="128_64",) # default to be 1 hidden of node_dim
parser.add_argument('--concat_states', action='store_true', default=False)
parser.add_argument('--norm_embs', type=str, default=None,) # default to be all truetrain_mdi.py
parser.add_argument('--aggr', type=str, default='mean',)
parser.add_argument('--node_dim', type=int, default=128)
parser.add_argument('--edge_dim', type=int, default=128)
parser.add_argument('--edge_mode', type=int, default=1)  # 0: use it as weight; 1: as input to mlp
parser.add_argument('--gnn_activation', type=str, default='relu')
parser.add_argument('--impute_hiddens', type=str, default='128_64')
parser.add_argument('--impute_activation', type=str, default='relu')
parser.add_argument('--epochs', type=int, default=epochs)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--opt_scheduler', type=str, default='none')
parser.add_argument('--opt_restart', type=int, default=0)
parser.add_argument('--opt_decay_step', type=int, default=1000)
parser.add_argument('--opt_decay_rate', type=float, default=0.9)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--lr', type=float, default=lr)
parser.add_argument('--known', type=float, default=1-unlab_train) # 1 - edge dropout rate
parser.add_argument('--auto_known', action='store_true', default=False)
parser.add_argument('--loss_mode', type=int, default = 0) # 0: loss on all train edge, 1: loss only on unknown train edge
parser.add_argument('--valid', type=float, default=0.05) # valid-set ratio
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--log_dir', type=str, default=log_dir)
parser.add_argument('--save_model', action='store_true', default=True)
parser.add_argument('--save_prediction', action='store_true', default=False)
parser.add_argument('--transfer_dir', type=str, default=None)
parser.add_argument('--transfer_extra', type=str, default='')
parser.add_argument('--mode', type=str, default='train') # debug
subparsers = parser.add_subparsers()
add_uji_subparser(subparsers)
args = parser.parse_args()
print(args)

# select device
if torch.cuda.is_available():
    cuda = auto_select_gpu()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    device = torch.device('cuda:{}'.format(cuda))
else:
    print('Using CPU')
    device = torch.device('cpu')

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)

if args.domain == 'uji':
    from uji.uji_data_mdi import load_data_mdi
    data, df = load_data_mdi(args)

log_path = './{}/test/{}/{}/feat_imp/'.format(args.domain,args.data,args.log_dir)
os.makedirs(log_path)

cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(osp.join(log_path, 'cmd_input.txt'), 'a') as f:
    f.write(str(args))

train_gnn_mdi(data, args, log_path, df, device)

result_path = '{}/test/{}/{}/feat_imp/result.pkl'.format(args.domain,args.data,args.log_dir)



# Using prediction from feature imputation for label prediction of test dataset

parser = argparse.ArgumentParser()
parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE')
parser.add_argument('--post_hiddens', type=str, default="128_64",) # default to be 1 hidden of node_dim
parser.add_argument('--concat_states', action='store_true', default=False)
parser.add_argument('--norm_embs', type=str, default=None,) # default to be all true
parser.add_argument('--aggr', type=str, default='mean',)
parser.add_argument('--node_dim', type=int, default=128)
parser.add_argument('--edge_dim', type=int, default=128)
parser.add_argument('--edge_mode', type=int, default=1)  # 0: use it as weight 1: as input to mlp
parser.add_argument('--gnn_activation', type=str, default='relu')
parser.add_argument('--impute_hiddens', type=str, default='128_64')
parser.add_argument('--impute_activation', type=str, default='relu')
parser.add_argument('--predict_hiddens', type=str, default='')
parser.add_argument('--epochs', type=int, default=epochs)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--opt_scheduler', type=str, default='none')
parser.add_argument('--opt_restart', type=int, default=0)
parser.add_argument('--opt_decay_step', type=int, default=1000)
parser.add_argument('--opt_decay_rate', type=float, default=0.9)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--lr', type=float, default=lr)
parser.add_argument('--known', type=float, default=0.8) # 1 - edge dropout rate
parser.add_argument('--valid', type=float, default=0.) # valid-set ratio
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--log_dir', type=str, default=log_dir)
subparsers = parser.add_subparsers()
add_uji_subparser(subparsers)
args = parser.parse_args()
print(args)

# select device
if torch.cuda.is_available():
    cuda = auto_select_gpu()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    device = torch.device('cuda:{}'.format(cuda))
else:
    print('Using CPU')
    device = torch.device('cpu')

if args.domain == 'uji':
    from uji.uji_data import load_data
    data = load_data(args, y_mdi = True, log_dir = log_dir)

log_path = './{}/test/{}/{}/grape_imputed/grape/'.format(args.domain,args.data,args.log_dir)
os.makedirs(log_path)

cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(osp.join(log_path, 'cmd_input.txt'), 'a') as f:
    f.write(str(args))

train_gnn_y(data, args, log_path, device)


log_path = '{}/test/{}/{}/grape_imputed/kNN/'.format(args.domain,args.data,args.log_dir)
os.makedirs(log_path)
train_knn(args, both = True, log_path=log_path, result_path = result_path)

log_path = '{}/test/{}/{}/grape_imputed/MLP/'.format(args.domain,args.data,args.log_dir)
os.makedirs(log_path)
train_mlp(args, both = True, log_path=log_path, result_path = result_path)


log_path = './{}/test/{}/{}/grape_imputed/DEC/'.format(args.domain,args.data,args.log_dir)
os.makedirs(log_path)
train_dec(args, both = True, log_path=log_path, result_path = result_path)





# Check the accuracy without feature imputation dataset


if args.domain == 'uji':
    from uji.uji_data import load_data
    data = load_data(args, log_dir = log_dir)

log_path = './{}/test/{}/{}/mean_imputed/grape/'.format(args.domain,args.data,args.log_dir)
os.makedirs(log_path)

cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(osp.join(log_path, 'cmd_input.txt'), 'a') as f:
    f.write(str(args))

train_gnn_y(data, args, log_path, device)


log_path = './{}/test/{}/{}/mean_imputed/kNN/'.format(args.domain,args.data,args.log_dir)
os.makedirs(log_path)
train_knn(args, both = True, log_path=log_path, result_path = result_path)

log_path = './{}/test/{}/{}/mean_imputed/MLP/'.format(args.domain,args.data,args.log_dir)
os.makedirs(log_path)
train_mlp(args, both = True, log_path=log_path, result_path = result_path)

log_path = './{}/test/{}/{}/mean_imputed/DEC/'.format(args.domain,args.data,args.log_dir)
os.makedirs(log_path)
train_dec(args, both = True, log_path=log_path, result_path = result_path)