import pandas as pd


import matplotlib.pyplot as plt

import pandas as pd
import os.path as osp
import inspect

import torch
import pickle
from utils.plot_utils import plot_curve

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch

import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU

import torch.nn.functional as F
import torch.optim as optim

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def train_mlp(args, both = True, log_path = None, result_path=None):
    # uji_path = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
    uji_path = "uji"

    building_id = 1
    floor= 1
    
    df_val = pd.read_csv(uji_path+'/raw_data/{}/data/validationData.csv'.format(args.data))
    
    df_val = df_val.loc[df_val['BUILDINGID'] == building_id]
    df_val = df_val.loc[df_val['FLOOR'] == floor]
    
    if both:

        df_val1 = df_val.iloc[: , 0:520]
        df_val1[['x_scaled', 'y_scaled']] = df_val[['x_scaled', 'y_scaled']]
        
        df_val_x = df_val1.iloc[: , 0:520]
        df_val_y = df_val1.iloc[: , -2:]
        
        with open(result_path, 'rb') as f:
            obj = pickle.load(f)

        df_train = pd.DataFrame(obj['filled_data'], columns=df_val1.columns)
        
        df_train_x = df_train.iloc[: , 0:520]
        df_train_y = df_train.iloc[: , -2:]

    else:
        df_val_x = df_val.iloc[: , 0:520]
        df_val_y = df_val.iloc[: , -2:]
        df_train = pd.read_csv(uji_path+'/raw_data/{}/data/trainingData.csv'.format(args.data))

        df_train = df_train.loc[df_train['BUILDINGID'] == building_id]
        df_train = df_train.loc[df_train['FLOOR'] == floor]

        with open(result_path, 'rb') as f:
            obj = pickle.load(f)

        mask = obj["train_edge_mask"].view(-1, 522)[:, -1]
        df_train = df_train[mask.numpy()]

        df_train_x = df_train.iloc[: , 0:520]
        df_train_y = df_train.iloc[: , -2:]


    norm_scl = MinMaxScaler()
    # std_scl = StandardScaler()

    norm_scl.fit(df_train_x)
    # std_scl.fit(df_train_y)

    df_train_x = norm_scl.transform(df_train_x)
    # df_train_y = std_scl.transform(df_train_y)

    x_train_tensor = torch.tensor(df_train_x).float()
    y_train_tensor = torch.tensor(df_train_y.values).float()

    df_val_x = norm_scl.transform(df_val_x)
    # df_val_y = std_scl.transform(df_val_y)

    x_val_tensor = torch.tensor(df_val_x).float()
    y_val_tensor = torch.tensor(df_val_y.values).float()

    best_k = 0
    best_error = 1000000

    obj = dict()
    obj['dist_error'] = []

    h_list = [(32), (64), (64, 8), (256, 16), (256, 64, 8), (512, 64), (512, 256, 64), (1024, 512, 256), (1024, 16), (2048, 512, 16), (2048, 64), (4096, 512, 32), (4096, 64)]
    for i, layer in enumerate(h_list):
        print(layer)
        mlp = MLPRegressor(hidden_layer_sizes=layer, max_iter=10000, random_state=0)

        # Train the MLP regressor
        mlp.fit(x_train_tensor, y_train_tensor)

        # Test the MLP regressor
        score = mlp.score(x_val_tensor, y_val_tensor)
        print(f"Test score: {score:.2f}")
        error = torch.mean(torch.linalg.norm((torch.tensor(mlp.predict(x_val_tensor)) - y_val_tensor), 2, dim=1))
        obj['dist_error'].append(error.item())
        print(error, "mtrs")

    plot_curve(obj, log_path+'curves.png',keys=None, clip=True, label_min=True, label_end=True)