import random
import numpy as np
import pandas as pd
import os.path as osp
import inspect
import pickle
from utils.plot_utils import plot_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import DynamicEdgeConv, MessagePassing, knn_graph
from torch.nn import Sequential as Seq, Linear, ReLU

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.utils import get_known_mask_val

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_dec(args, both = False, log_path=None, result_path=None) :
    # uji_path = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
    uji_path = "uji"
            
    building_id = 1
    floor= 1
    
    df_val = pd.read_csv(uji_path+'/raw_data/{}/data/validationData.csv'.format(args.data))
    
    # df_val = df_val.loc[df_val['BUILDINGID'] == building_id]
    df_val = df_val.loc[df_val['FLOOR'] == floor]
    
    if both:

        df_val1 = df_val.iloc[: , 0:520]
        df_val1[['x_scaled', 'y_scaled']] = df_val[['x_scaled', 'y_scaled']]
        test_num = len(df_val1)
        
        with open(result_path, 'rb') as f:
            obj = pickle.load(f)

        df_train = pd.DataFrame(obj['filled_data'], columns=df_val1.columns)

        df = pd.concat([df_train, df_val1])

        df_X = df.iloc[: , 0:520]
        df_y = df[['x_scaled', 'y_scaled']]

    else:
        df_train = pd.read_csv(uji_path+'/raw_data/{}/data/trainingData.csv'.format(args.data))

        # df_train = df_train.loc[df_train['BUILDINGID'] == building_id]
        df_train = df_train.loc[df_train['FLOOR'] == floor]

        with open(result_path, 'rb') as f:
            obj = pickle.load(f)

        mask = obj["train_edge_mask"].view(-1, 522)[:, -1]
        df_train = df_train[mask.numpy()]

        test_num = len(df_val)
        df = pd.concat([df_train, df_val])

        df_X = df.iloc[: , 0:520]
        df_y = df[['x_scaled', 'y_scaled']]

    norm_scl = MinMaxScaler()
    # std_scl = StandardScaler()

    norm_scl.fit(df_X)
    # std_scl.fit(df_y)

    df_x = norm_scl.transform(df_X)
    # df_y = std_scl.transform(df_y)

    # Define the node features
    x = torch.tensor(df_x, dtype=torch.float).float()

    # Define the node labels
    y = torch.tensor(df_y.values, dtype=torch.float).float()

    # Create the Data object
    data = Data(x=x, y=y)
    data = data.to(device)

    train_mask = get_known_mask_val(None, y.shape[0], test_num)

    data.train_mask = train_mask
    data.test_mask = ~train_mask

    class EdgeConv(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super().__init__(aggr='mean')
            self.mlp = Seq(Linear(2 * in_channels, out_channels), ReLU(), Linear(out_channels, out_channels))

        def forward(self, x, edge_index):
            return self.propagate(edge_index, x=x)

        def message(self, x_i, x_j):
            tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
            return self.mlp(tmp)
    
    class DynamicEdgeConv(EdgeConv):
        def __init__(self, in_channels, out_channels, k=5):
            super().__init__(in_channels, out_channels)
            self.k = k

        def forward(self, x, batch=None):
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
            return super().forward(x, edge_index)
    
    class DEC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = DynamicEdgeConv(data.num_features, 1024, k=5)
            self.conv2 = DynamicEdgeConv(1024, 512, k=4)
            self.conv3 = DynamicEdgeConv(512, 128, k=3)
            self.fc1 = nn.Linear(128, 32)
            self.fc2 = nn.Linear(32, 2)
            self.dropout = nn.Dropout(p=0.5)

        def forward(self, data):
            x = data.x
            x = self.conv1(x)
            x = F.relu(x)
            # x = F.dropout(x, training=self.training)
            x = self.dropout(x)
            x = self.conv2(x)
            x = F.relu(x)
            # x = F.dropout(x, training=self.training)
            x = self.dropout(x)
            x = self.conv3(x)
            x = F.relu(x)
            # x = F.dropout(x, training=self.training)
            x = self.dropout(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)


            return x
        
    model = DEC().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    val_losses = []

    obj = dict()
    
    # while True:
    for epoch in range(5000):
        model.train()
        optimizer.zero_grad()
        out = model(data)

        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        
        model.eval()
        pred = model(data)
        # val_loss = F.mse_loss(pred[data.test_mask], data.y[data.test_mask])
        val_loss = torch.mean(torch.linalg.norm((pred[data.test_mask] - data.y[data.test_mask]), 2, dim=1))
        val_losses.append(val_loss.item())
        
        if (epoch+1) % 5 == 0:
            print('Epoch: ', epoch+1, 'Loss: ', loss.item(), 'Val Loss: ', val_loss.item())
            
            obj['train_rmse'] = losses
            obj['test_dist_error'] = val_losses
            plot_curve(obj, log_path+'curves.png',keys=None, clip=True, label_min=True, label_end=True)


    obj['train_rmse'] = losses
    obj['test_dist_error'] = val_losses

    plot_curve(obj, log_path+'curves.png',keys=None, clip=True, label_min=True, label_end=True)