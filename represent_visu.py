from cProfile import label
from cgi import test
from pyexpat import features
from statistics import mode
from tkinter.tix import Tree
import torch
import numpy as np
import math
import torch.utils.data as utils
from sklearn.metrics import accuracy_score
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, precision_score
from itertools import cycle
import os
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE

RATE = 70
TASK = 'GRU-T'
CKPT_PATH = r'/media/liu/Data/Project/Python/ICURelated/NeuralODE/GEUD-ODE-gao/result_sample_70/GRU ODE DH/model_best_VAL.pkl'
rootpath = r'/media/liu/Data/DataSet/Physionet2012/Sampled'
save_path = f'/media/liu/Data/Project/Python/ICURelated/NeuralODE/GEUD-ODE-gao/represent_sample_{str(RATE)}/{TASK}'
os.makedirs(save_path, exist_ok=True)
#all_x_add = np.load(rootpath + 'input/all_x_add.npy', allow_pickle=True)
#dataset = np.load(rootpath + 'input/dataset.npy', allow_pickle=True)
dataset = np.load(os.path.join(rootpath, f'dataset_{str(RATE)}.npy'), allow_pickle=True)
dt = np.load(os.path.join(rootpath, f'dt_{str(RATE)}.npy'), allow_pickle=True)
# 0:death 1:length of stay(<3) 2:Cardical 3:Surgery
y = np.load(os.path.join(rootpath, 'y.npy'), allow_pickle=True)
y1 = y[:,0:1] 

class GRUD_ODECell(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    
    self.input_size = input_size
    self.hidden_size = hidden_size
    
    self.inputzeros = torch.autograd.Variable(torch.zeros(input_size))
    self.hiddenzeros = torch.autograd.Variable(torch.zeros(hidden_size))
    
    self.w_dg_x = torch.nn.Parameter(torch.Tensor(input_size))
    self.w_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size,input_size))
    self.b_dg_x = torch.nn.Parameter(torch.Tensor(input_size))
    self.b_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size))

    self.lin_xh = torch.nn.Linear(input_size, hidden_size, bias=True)
    self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=True)
    self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=True)
    self.lin_hu = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_su = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_sz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_sr = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    self.lin_mu = torch.nn.Linear(input_size, hidden_size, bias=False)
    self.lin_mz = torch.nn.Linear(input_size, hidden_size, bias=False)
    self.lin_mr = torch.nn.Linear(input_size, hidden_size, bias=False)

    # self.w_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size))
    # # z
    # self.w_xz = torch.nn.Parameter(torch.Tensor(input_size))
    # self.w_hz = torch.nn.Parameter(torch.Tensor(hidden_size))
    # self.w_mz = torch.nn.Parameter(torch.Tensor(input_size))

    # # r
    # self.w_xr = torch.nn.Parameter(torch.Tensor(input_size))
    # self.w_hr = torch.nn.Parameter(torch.Tensor(hidden_size))
    # self.w_mr = torch.nn.Parameter(torch.Tensor(input_size))

    # # h_tilde
    # self.w_xh = torch.nn.Parameter(torch.Tensor(input_size))
    # self.w_hh = torch.nn.Parameter(torch.Tensor(hidden_size))
    # self.w_mh = torch.nn.Parameter(torch.Tensor(input_size))

    # # bias
    # self.b_z = torch.nn.Parameter(torch.Tensor(hidden_size))
    # self.b_r = torch.nn.Parameter(torch.Tensor(hidden_size))
    # self.b_h = torch.nn.Parameter(torch.Tensor(hidden_size))

  def forward(self, h, x, m, d, prex, mean, dh):
    gamma_x = torch.exp(-torch.max(self.inputzeros, (self.w_dg_x*d + self.b_dg_x)))
    gamma_h = torch.exp(-torch.max(self.hiddenzeros, (torch.matmul(self.w_dg_h, d) + self.b_dg_h)))
    # gamma_h = torch.exp(-torch.max(self.hiddenzeros, (self.w_dg_h*d + self.b_dg_h)))
    x = m * x + (1 - m) * (gamma_x * prex + (1 - gamma_x) * mean)
    # x = m * x + (1 - m) * prex
    # x = m * x + (1 - m) * mean
    h = gamma_h * h
    
    r = torch.sigmoid(self.lin_xr(x) + self.lin_hr(h) + self.lin_sr(dh) +self.lin_mr(m))
    z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h) + self.lin_sz(dh) + self.lin_mz(m))
    u = torch.tanh(self.lin_xh(x) + self.lin_hu(r * h) + self.lin_su(dh) + self.lin_mu(m))
    
    # r = torch.sigmoid(self.lin_xr(x) + self.lin_hr(h) + self.lin_mr(m))
    # z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h) + self.lin_mz(m))
    # u = torch.tanh(self.lin_xh(x) + self.lin_hu(r * h) + self.lin_mu(m))


    h_post = (1-z) * h + z * u
    # h_post = h_post * gamma_h
    dh = z * (u - h)

    # gamma_h gru_ode_t
    # dh = dh * gamma_h
    return h_post, dh, x

class GRUD_ODE(torch.nn.Module):  
  def __init__(self, input_size, hidden_size, output_size, dropout_type, dropout):
    super().__init__()

    self.dropout_type = dropout_type
    self.dropout = dropout
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.cell = GRUD_ODECell(input_size, hidden_size)
    # self.lin_1 = torch.nn.Linear(hidden_size[0], hidden_size[1], bias=True)
    # self.lin_2 = torch.nn.Linear(hidden_size[1], output_size, bias=True)
    # self.lin = torch.nn.Sequential(
    #             self.lin_1,
    #             torch.nn.ReLU(),
    #             self.lin_2
    #             )
    self.lin = torch.nn.Linear(hidden_size, output_size, bias=True)
    self.reset_parameters()
  
  def reset_parameters(self):
    #stdv = 1.0 / math.sqrt(self.hidden_size)
    #for weight in self.parameters():
    #  torch.nn.init.uniform_(weight, -stdv, stdv)
    for params in self.parameters():
      torch.nn.init.normal_(params, mean=0, std=0.1)

  def forward(self, input, dt):
    X = torch.squeeze(input[0]) 
    Mask = torch.squeeze(input[1]) 
    Delta = torch.squeeze(input[2]) 
    #dt = torch.squeeze(dt) 
    h = torch.autograd.Variable(torch.zeros(self.hidden_size))
    dh = torch.autograd.Variable(torch.zeros(self.hidden_size))
    prex = torch.autograd.Variable(torch.zeros(self.input_size))
    mean = torch.squeeze(torch.sum(X,1))/(1e-6+torch.squeeze(torch.sum((Mask!=0),1)))
    for layer in range(X.shape[1]):
      if dt[layer]==0:
        break

      x = torch.squeeze(X[:,layer])
      m = torch.squeeze(Mask[:,layer])
      d = torch.squeeze(Delta[:,layer])
      if self.dropout == 0:
        h_post, dh, prex = self.cell(h, x, m, d, prex, mean, dh)
        h = h + dt[layer]*dh
        # h = h_post
        # h = h + dh
      elif self.dropout_type == 'Moon':
        h_post, dh, prex = self.cell(h, x, m, d, prex,  mean, dh)
        h = h + dt[layer]*dh
        # h = h_post
        # h = h + dh
        dropout = torch.nn.Dropout(p=self.dropout)
        h = dropout(h)
      elif self.dropout_type == 'Gal':
        dropout = torch.nn.Dropout(p=self.dropout)
        h = dropout(h)
        h_post, dh, prex = self.cell(h, x, m, d, prex,  mean, dh)
        h = h + dt[layer]*dh
        # h = h_post
        # h = h + dh
      elif self.dropout_type == 'mloss':
        dropout = torch.nn.Dropout(p=self.dropout)
        h_post, dh, prex = self.cell(h, x, m, d, prex, mean,  dh)
        dh = dropout(dh)
        h = h + dt[layer]*dh
        # h = h_post
        # h = h + dh    
    output = self.lin(h)      
    output = torch.sigmoid(output)
    return output, h


def inference(model, dataloader):
    features_list = []
    labels_list = []
    model.eval()
    for train_data, train_label, dt in tqdm(dataloader):
      train_data = torch.squeeze(train_data)
      train_label = torch.squeeze(train_label)
      dt = torch.squeeze(dt) 
      y_pred, features = model(train_data, dt)
      features = features.detach().numpy()
      features_list.append(features)
      labels_list.append(train_label.cpu().numpy())
    np.save(os.path.join(save_path, 'features.npy'),np.array(features_list))
    np.save(os.path.join(save_path, 'labels.npy'),np.array(labels_list))

def t_sne_visu():
    features = np.load(os.path.join(save_path, 'features.npy'))
    labels = np.load(os.path.join(save_path, 'labels.npy')).tolist()
    x_embedded = TSNE(n_components=2, init='random').fit_transform(features)
    phe_dict = {0: 'Mor', 1: 'Los', 2: 'Car', 3: 'Sur'}
    df_list = []
    for i in range(4):
        label_bin = list(map(lambda x: 'Postive' if x[i] > 0 else 'Negative', labels))
        df = pd.DataFrame({'X': x_embedded[:, 0].tolist(), 'Y': x_embedded[:, 1].tolist(), 'Label': label_bin, 'Phenotyping': [phe_dict[i]]*len(label_bin)})
        df_list.append(df)
    df = pd.concat(df_list)
    g = sns.FacetGrid(df, col='Phenotyping', col_wrap=2, hue='Label')
    g.map(sns.scatterplot, 'X', 'Y', alpha=0.7)
    g.add_legend()
    plt.show()

def t_sne_visu_1():
    features = np.load(os.path.join(save_path, 'features.npy'))
    labels = np.load(os.path.join(save_path, 'labels.npy')).tolist()
    x_embedded = TSNE(n_components=2, init='random').fit_transform(features)
    label_bin = list(map(lambda x: str(x), labels))
    df = pd.DataFrame({'X': x_embedded[:, 0].tolist(), 'Y': x_embedded[:, 1].tolist(), 'Label': label_bin})
    sns.scatterplot(data=df, x='X', y='Y', hue='Label')
    plt.show()


def data_dataloader(dataset, outcomes, dt, \
                    train_proportion = 0.7, dev_proportion = 0.15, test_proportion=0.15):
    
  train_index = int(np.floor(dataset.shape[0] * train_proportion))
  val_index = int(np.floor(dataset.shape[0] * dev_proportion))
  val_index = train_index + val_index
  # split dataset to tarin/dev/test set
  train_data, train_label = dataset[:train_index,:,:,:], outcomes[:train_index,:]
  dev_data, dev_label = dataset[train_index: val_index,:,:,:], outcomes[train_index:val_index,:]  
  test_data, test_label = dataset[val_index: ,:,:,:], outcomes[val_index: ,:]  
  train_dt, dev_dt, test_dt = dt[:train_index,:], dt[train_index:val_index,:], dt[val_index: ,:]
    
  # ndarray to tensor
  train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
  dev_data, dev_label = torch.Tensor(dev_data), torch.Tensor(dev_label)
  test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)
  train_dt, dev_dt, test_dt = torch.Tensor(train_dt), torch.Tensor(dev_dt), torch.Tensor(test_dt)
  
  # tensor to dataset
  train_dataset = utils.TensorDataset(train_data, train_label, train_dt)
  dev_dataset = utils.TensorDataset(dev_data, dev_label, dev_dt)
  test_dataset = utils.TensorDataset(test_data, test_label, test_dt)
  
  # dataset to dataloader 
  train_dataloader = utils.DataLoader(train_dataset)
  dev_dataloader = utils.DataLoader(dev_dataset)
  test_dataloader = utils.DataLoader(test_dataset)
  
  return train_dataloader, dev_dataloader, test_dataloader

def inference_entrance():
  input_size = 33 
  hidden_size = 64 
  output_size = 4
  
  #dropout_type : Moon, Gal, mloss
  model = GRUD_ODE(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout_type='Moon', dropout=0.0)
  ckpt = torch.load(CKPT_PATH)
  model.load_state_dict(ckpt)
  
  train_dataloader, dev_dataloader, test_dataloader = data_dataloader(dataset, y, dt)
  inference(model, dev_dataloader)


if __name__ == '__main__':
    # inference_entrance()
    # t_sne_visu()
    t_sne_visu_1()