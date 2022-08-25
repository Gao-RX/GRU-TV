from cgi import test
from statistics import mode
from tkinter.tix import Tree
import torch
import numpy as np
import math
import torch.utils.data as utils
from sklearn.metrics import accuracy_score
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, precision_score
from itertools import cycle
import os
from tqdm import tqdm

from sklearn.metrics import roc_auc_score

result_path = r'/media/liu/Data/Project/Python/ICURelated/NeuralODE/GEUD-ODE-gao/result_sample_50'
TASK_NAME = 'GRU'
rootpath = r'/media/liu/Data/DataSet/Physionet2012/Sampled'
result_path = os.path.join(result_path, TASK_NAME)
os.makedirs(result_path, exist_ok=True)
#all_x_add = np.load(rootpath + 'input/all_x_add.npy', allow_pickle=True)
#dataset = np.load(rootpath + 'input/dataset.npy', allow_pickle=True)
dataset = np.load(os.path.join(rootpath, 'dataset_50.npy'), allow_pickle=True)
dt = np.load(os.path.join(rootpath, 'dt_50.npy'), allow_pickle=True)
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

  def forward(self, h, x, m, d, prex, mean):
    gamma_x = torch.exp(-torch.max(self.inputzeros, (self.w_dg_x*d + self.b_dg_x)))
    gamma_h = torch.exp(-torch.max(self.hiddenzeros, (torch.matmul(self.w_dg_h, d) + self.b_dg_h)))
    # gamma_h = torch.exp(-torch.max(self.hiddenzeros, (self.w_dg_h*d + self.b_dg_h)))
    # x = m * x + (1 - m) * (gamma_x * prex + (1 - gamma_x) * mean)
    x = m * x + (1 - m) * prex
    # x = m * x + (1 - m) * mean
    # h = gamma_h * h
    
    r = torch.sigmoid(self.lin_xr(x) + self.lin_hr(h) + self.lin_mr(m))
    z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h) + self.lin_mz(m))
    u = torch.tanh(self.lin_xh(x) + self.lin_hu(r * h) + self.lin_mu(m))
    # z = torch.sigmoid((self.w_xz*x + self.w_hz*h + self.w_mz*m + self.b_z))
    # r = torch.sigmoid((self.w_xr*x + self.w_hr*h + self.w_mr*m + self.b_r))
    # u = torch.tanh((self.w_xh*x + self.w_hh*(r * h) + self.w_mh*m + self.b_h))
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
    prex = torch.autograd.Variable(torch.zeros(self.input_size))
    mean = torch.squeeze(torch.sum(X,1))/(1e-6+torch.squeeze(torch.sum((Mask!=0),1)))
    for layer in range(X.shape[1]):
      if dt[layer]==0:
        break

      x = torch.squeeze(X[:,layer])
      m = torch.squeeze(Mask[:,layer])
      d = torch.squeeze(Delta[:,layer])
      if self.dropout == 0:
        h_post, dh, prex = self.cell(h, x, m, d, prex, mean)
        # h = h + dt[layer]*dh
        h = h_post
        # h = h + dh
      elif self.dropout_type == 'Moon':
        h_post, dh, prex = self.cell(h, x, m, d, prex, mean)
        # h = h + dt[layer]*dh
        h = h_post
        # h = h + dh
        dropout = torch.nn.Dropout(p=self.dropout)
        h = dropout(h)
      elif self.dropout_type == 'Gal':
        dropout = torch.nn.Dropout(p=self.dropout)
        h = dropout(h)
        h_post, dh, prex = self.cell(h, x, m, d, prex, mean)
        # h = h + dt[layer]*dh
        h = h_post
        # h = h + dh
      elif self.dropout_type == 'mloss':
        dropout = torch.nn.Dropout(p=self.dropout)
        h_post, dh, prex = self.cell(h, x, m, d, prex, mean)
        dh = dropout(dh)
        # h = h + dt[layer]*dh
        h = h_post
        # h = h + dh    
    output = self.lin(h)      
    output = torch.sigmoid(output)
    return output


def fit(model, criterion, learning_rate,\
        train_dataloader, dev_dataloader, test_dataloader,\
        learning_rate_decay, n_epochs):
  epoch_losses = []
  best_val = -1
  best_test = -1
  no_increase = 0
  for epoch in range(n_epochs):   
    if learning_rate_decay != 0:
      if  epoch % learning_rate_decay == 0:
        learning_rate = learning_rate/2
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif learning_rate_decay == 0:
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
      
    # train the model
    losses = []
    model.train()
    for train_data, train_label, dt in tqdm(train_dataloader):
      optimizer.zero_grad() 
      train_data = torch.squeeze(train_data)
      train_label = torch.squeeze(train_label)
      dt = torch.squeeze(dt) 
      y_pred = model(train_data, dt)

      #pred.append(torch.argmax(y_pred,dim=0).item())
      # pred.append(y_pred.item())
      # label.append(train_label.item())
      #loss = criterion(y_pred.view(1,-1), train_label.long())
      loss = criterion(y_pred, train_label)
      # acc.append(
      #     accuracy_score([train_label.item()], [(y_pred.item()>0.5)+0])
      # )
      losses.append(loss.item())

      loss.backward()
      optimizer.step()
    torch.save(model.state_dict(), os.path.join(result_path, 'last.pkl'))
    print('='*35+str(epoch)+'='*35)
    need_test = False
    auc_mean = test_model(model, dev_dataloader, 'VAL', best_val)
    if auc_mean > best_val:
      best_val = auc_mean
      no_increase = 0
      need_test = True
      print('='*35+'IMPROVED'+'='*35)
    else:
      no_increase += 1

    if no_increase >= 3 and epoch > 25:
      break

    if need_test:
        auc_mean = test_model(model, test_dataloader, 'TEST', best_test)
        if auc_mean > best_test:
          best_test = auc_mean
          print('='*35+'NEW BEST'+'='*35)

def test_model(model, dev_dataloader, task, best_per):
    losses = []
    pred, label = list(), list()
    model.eval()
    for dev_data, dev_label, dt in tqdm(dev_dataloader):
      dev_data = torch.squeeze(dev_data)
      dev_label = torch.squeeze(dev_label)
      dt = torch.squeeze(dt)

      y_pred = model(dev_data, dt)
      
      #pred.append(torch.argmax(y_pred,dim=0).item())
      pred.append(y_pred.detach().numpy().tolist())
      label.append(dev_label.detach().numpy().tolist())
      #loss = criterion(y_pred.view(1,-1), train_label.long())AUC
      loss = criterion(y_pred, dev_label)
      # acc.append(
      #     accuracy_score([dev_label.item()], [(y_pred.item()>0.5)+0])
      # )
      losses.append(loss.item())
          
    # dev_acc = np.mean(acc)
    dev_loss = np.mean(losses)
    
    pred = np.asarray(pred)
    label = np.asarray(label)

    auc_mean, roc_auc = get_performance(predicts=pred, labels=label, best_per=best_per, save_path=result_path, task=task)
    if auc_mean > best_per:
      best_per = auc_mean
      np.save(os.path.join(result_path, 'pred_best_{}.npy'.format(task)) , pred)
      np.save(os.path.join(result_path, 'label_best_{}.npy'.format(task)), label)
      torch.save(model.state_dict(), os.path.join(result_path, 'model_best_{}.pkl'.format(task)))
    
    auc_score = roc_auc_score(label, pred)
    auc_score_1 = roc_auc_score(label[:,0], pred[:,0])
    auc_score_2 = roc_auc_score(label[:,1], pred[:,1])
    auc_score_3 = roc_auc_score(label[:,2], pred[:,2])
    auc_score_4 = roc_auc_score(label[:,3], pred[:,3])
    
    print('='*35+task+'='*35)
    print("AUC score: {:.4f}".format(
       auc_score))
    print("ortality: {:.4f}, Length of Stay: {:.4f}, Cardiac: {:.4f}, Surgery: {:.4f}".format(
        auc_score_1, auc_score_2, auc_score_3, auc_score_4))
    print("Mean: {:.4f}, Mortality: {:.4f}, Length of Stay: {:.4f}, Cardiac: {:.4f}, Surgery: {:.4f}".format(
        auc_mean, roc_auc['class_0'], roc_auc['class_1'], roc_auc['class_2'], roc_auc['class_3']))
    return auc_mean


def get_performance(predicts, labels, best_per, save_path, task):
    num_class = 4
    if not isinstance(predicts, np.ndarray):
        predicts = np.array(predicts)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_total = 0
    # predicts_result = np.zeros(predicts.shape)
    # predicts_result[predicts>=0.5]=1
    labels[labels<0.5]=0
    labels[labels!=0]=1
    for i in range(num_class):
        class_name = 'class_{}'.format(str(i))
        fpr[class_name], tpr[class_name], _ = roc_curve(labels[:, i], predicts[:, i])
        roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])
        auc_total += roc_auc[class_name]
    
    auc_mean = auc_total / num_class
    roc_auc['macro']= roc_auc_score(labels, predicts, average='macro')
    roc_auc['micro'] = roc_auc_score(labels, predicts, average='micro')

    colors = cycle(['blue', 'red', 'green', 'black'])

    plt.figure(figsize=(40, 25))
    for i, color in zip(fpr.keys(), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                label='class {0} ({1:0.5f})'
                ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC')
    plt.legend(loc="lower right")
    if auc_mean > best_per:
        plt.savefig(os.path.join(save_path, '{}_best.png'.format(task)))
    print('*'*40+task+'*'*40)
    # print('FPR:{}'.format('  '.join([str(round(fpr[x], 2)) for x in fpr.keys()])))
    # print('TPR:{}'.format('  '.join([str(round(tpr[x], 2)) for x in tpr.keys()])))
    print('AUC:{}'.format('  '.join([str(round(roc_auc[x], 2)) for x in roc_auc.keys()])))
    print('AUC_MEAN:{}'.format(auc_mean))
    plt.close()
    return auc_mean, roc_auc

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

def load_pickle():
  a = np.load(r'E:\DataSet\Physionet2012\Sampled\dataset_100.npy')
  print(a.shape)

if __name__ == '__main__':    
  input_size = 33 
  hidden_size = 10 
  output_size = 4
  
  #dropout_type : Moon, Gal, mloss
  model = GRUD_ODE(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout_type='Moon', dropout=0.0)
  criterion = torch.nn.BCELoss()
  
  learning_rate = 0.01
  learning_rate_decay = 5
  n_epochs = 100
  train_dataloader, dev_dataloader, test_dataloader = data_dataloader(dataset, y, dt)
  fit(model, criterion, learning_rate,\
      train_dataloader, dev_dataloader, test_dataloader,\
      learning_rate_decay, n_epochs)
