import torch
import numpy as np
import math
import torch.utils.data as utils
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
# from keras.utils import to_categorical
from scipy.integrate import odeint
import matplotlib.pyplot as plt

root = ''
#all_x_add = np.load(rootpath + 'input/all_x_add.npy', allow_pickle=True)
#dataset = np.load(rootpath + 'input/dataset.npy', allow_pickle=True)
dataset = np.load(root+'dataset_100.npy', allow_pickle=True)
dt = np.load(root+'dt_100.npy', allow_pickle=True)
# 0:death 1:length of stay(<3) 2:Cardical 3:Surgery
y = np.load(root+'y.npy', allow_pickle=True)

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

  def forward(self, h, x, m, d, mean):
    gamma_x = torch.exp(-torch.max(self.inputzeros, (self.w_dg_x*d + self.b_dg_x)))
    gamma_h = torch.exp(-torch.max(self.hiddenzeros, (torch.matmul(self.w_dg_h, d) + self.b_dg_h)))
    # gamma_h = torch.exp(-torch.max(self.hiddenzeros, (self.w_dg_h*d + self.b_dg_h)))

    x = m * x + (1 - m) * (gamma_x * x + (1 - gamma_x) * mean)
    # h = gamma_h * h
    
    r = torch.sigmoid(self.lin_xr(x) + self.lin_hr(h) + self.lin_mr(m))
    z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h) + self.lin_mz(m))
    u = torch.tanh(self.lin_xh(x) + self.lin_hu(r * h) + self.lin_mu(m))
    # z = torch.sigmoid((self.w_xz*x + self.w_hz*h + self.w_mz*m + self.b_z))
    # r = torch.sigmoid((self.w_xr*x + self.w_hr*h + self.w_mr*m + self.b_r))
    # u = torch.tanh((self.w_xh*x + self.w_hh*(r * h) + self.w_mh*m + self.b_h))
    h_post = (1-z) * h + z * u
    
    dh = z * (u - h)
    dh = dh * gamma_h
    return dh, x

class GRUD_ODE(torch.nn.Module):  
  def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_type, dropout):
    super().__init__()

    self.dropout_type = dropout_type
    self.dropout = dropout
    self.num_layers = num_layers
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
    mean = torch.squeeze(torch.sum(X*Mask,1))/(1e-6+torch.squeeze(torch.sum((Mask!=0),1)))

    for layer in range(self.num_layers):
      if dt[layer]==0:
        break

      x = torch.squeeze(X[:,layer])
      m = torch.squeeze(Mask[:,layer])
      d = torch.squeeze(Delta[:,layer])
      if self.dropout == 0:
        dh, _ = self.cell(h, x, m, d, mean)
        h = h + dt[layer]*dh
        #h = h + dh
      elif self.dropout_type == 'Moon':
        dh, _ = self.cell(h, x, m, d, mean)
        h = h + dt[layer]*dh
        #h = h + dh
        dropout = torch.nn.Dropout(p=self.dropout)
        h = dropout(h)
      elif self.dropout_type == 'Gal':
        dropout = torch.nn.Dropout(p=self.dropout)
        h = dropout(h)
        dh, prex = self.cell(h, x, m, d, mean)
        h = h + dt[layer]*dh
        #h = h + dh
      elif self.dropout_type == 'mloss':
        dropout = torch.nn.Dropout(p=self.dropout)
        dh, prex = self.cell(h, x, m, d, mean)
        dh = dropout(dh)
        h = h + dt[layer]*dh
        #h = h + dh  

    output = self.lin(h)      
    output = torch.sigmoid(output)
    return output


def fit(model, criterion, learning_rate,\
        train_dataloader, dev_dataloader,\
        learning_rate_decay, n_epochs):
  epoch_losses = []
      
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


    for train_data, train_label, dt in train_dataloader:
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

    # train_acc = np.mean(acc)
    train_loss = np.mean(losses)
      
    # dev loss
    losses = []
    pred, label = list(), list()
    model.eval()
    for dev_data, dev_label, dt in dev_dataloader:
      dev_data = torch.squeeze(dev_data)
      dev_label = torch.squeeze(dev_label)
      dt = torch.squeeze(dt)

      y_pred = model(dev_data, dt)
      
      #pred.append(torch.argmax(y_pred,dim=0).item())
      pred.append(y_pred.detach().numpy().tolist())
      label.append(dev_label.detach().numpy().tolist())
      #loss = criterion(y_pred.view(1,-1), train_label.long())
      loss = criterion(y_pred, dev_label)
      # acc.append(
      #     accuracy_score([dev_label.item()], [(y_pred.item()>0.5)+0])
      # )
      losses.append(loss.item())
          
    # dev_acc = np.mean(acc)
    dev_loss = np.mean(losses)
    
    pred = np.asarray(pred)
    label = np.asarray(label)
    
    auc_score = roc_auc_score(label, pred)
    auc_score_1 = roc_auc_score(label[:,0], pred[:,0])
    auc_score_2 = roc_auc_score(label[:,1], pred[:,1])
    auc_score_3 = roc_auc_score(label[:,2], pred[:,2])
    auc_score_4 = roc_auc_score(label[:,3], pred[:,3])
    
    print("Epoch {}:\n Train loss: {:.4f}, Dev loss: {:.4f}\n AUC score: {:.4f}".format(
        epoch+1, train_loss, dev_loss, auc_score))
    print("Mortality: {:.4f}, Length of Stay: {:.4f}, Cardiac: {:.4f}, Surgery: {:.4f}".format(
        auc_score_1, auc_score_2, auc_score_3, auc_score_4))
  return epoch_losses     
                
def data_dataloader(dataset, outcomes, dt, \
                    train_proportion = 0.8, dev_proportion = 0.2):
    
  train_index = int(np.floor(dataset.shape[0] * train_proportion))
  
  # split dataset to tarin/dev/test set
  train_data, train_label = dataset[:train_index,:,:,:], outcomes[:train_index,:]
  dev_data, dev_label = dataset[train_index:,:,:,:], outcomes[train_index:,:]  
  train_dt, dev_dt = dt[:train_index,:], dt[train_index:,:]
    
  # ndarray to tensor
  train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
  dev_data, dev_label = torch.Tensor(dev_data), torch.Tensor(dev_label)
  train_dt, dev_dt = torch.Tensor(train_dt), torch.Tensor(dev_dt)
  
  # tensor to dataset
  train_dataset = utils.TensorDataset(train_data, train_label, train_dt)
  dev_dataset = utils.TensorDataset(dev_data, dev_label, dev_dt)
  
  # dataset to dataloader 
  train_dataloader = utils.DataLoader(train_dataset)
  dev_dataloader = utils.DataLoader(dev_dataset)
  
  return train_dataloader, dev_dataloader

if __name__ == '__main__':    
  input_size = 33 
  hidden_size = 10 
  output_size = 4
  TIME = 216
  
  #dropout_type : Moon, Gal, mloss
  model = GRUD_ODE(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=TIME, dropout_type='Moon', dropout=0.0)
  criterion = torch.nn.BCELoss()
  
  learning_rate = 0.01
  learning_rate_decay = 5
  n_epochs = 10
  train_dataloader, dev_dataloader = data_dataloader(dataset, y, dt, train_proportion=0.7, dev_proportion=0.3)
  epoch_losses = fit(model, criterion, learning_rate,\
                      train_dataloader, dev_dataloader,\
                      learning_rate_decay, n_epochs)