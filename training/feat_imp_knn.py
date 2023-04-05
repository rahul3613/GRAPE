import pandas as pd
import os.path as osp
import inspect
from utils.utils import get_known_mask_new

import torch
import pickle
from utils.plot_utils import plot_curve

from sklearn import neighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt


def train_knn_feat(args, prob, log_path=None):
    # uji_path = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
    uji_path = 'uji'

    building_id = 1
    floor= 1
    
    df = pd.read_csv(uji_path+'/raw_data/{}/data/trainingData.csv'.format(args.data))
    
    df = df.loc[df['BUILDINGID'] == building_id]
    df = df.loc[df['FLOOR'] == floor]

    df1 = df.iloc[: , 0:520]
    df1[['x_scaled', 'y_scaled']] = df[['x_scaled', 'y_scaled']]
    
    train_edge_mask = get_known_mask_new(prob, 5, df1.shape)
    
    mask = train_edge_mask.view(-1, 522)[:, -1]
    df_train = df1[mask.numpy()]
    df_val = df1[~mask.numpy()]
    
    df_val_x = df_val.iloc[: , 0:520]
    df_val_y = df_val.iloc[: , -2:]
    
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
    best_error = 10000

    obj = dict()
    obj['dist_error'] = []
    # obj['args'] = args
    obj['dist_error_val'] = []

    val_mask = torch.rand(len(x_train_tensor))
    mask = val_mask > 0.1


    for K in range(1, 20):
        model = neighbors.KNeighborsRegressor(n_neighbors = K)

        model.fit(x_train_tensor[mask], y_train_tensor[mask])  #fit the model

        pred=model.predict(x_train_tensor[~mask]) #make prediction on test set
        error_val = torch.mean(torch.linalg.norm((torch.tensor(pred) - y_train_tensor[~mask]), 2, dim=1))
        obj['dist_error_val'].append(error_val.item()) #store rmse values

        pred=model.predict(x_val_tensor) #make prediction on test set
        error = torch.mean(torch.linalg.norm((torch.tensor(pred) - y_val_tensor), 2, dim=1))
        obj['dist_error'].append(error.item()) #store rmse values
        
        if best_error > error_val:
            best_error = error_val.item()
            pred_test_best = pred
        
        print('Mean error in mtrs for k= ' , K , 'is:', error)

    plot_curve(obj, log_path+'curves.png',keys=None, clip=True, label_min=True, label_end=True)

    df1 = torch.tensor(df1.to_numpy()).float()
    df1[~train_edge_mask.view(-1, 522)] = torch.tensor(pred_test_best).float().view(-1)
    obj['filled_data'] = df1

    obj['train_edge_mask'] = train_edge_mask.view(-1, 522)
    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))