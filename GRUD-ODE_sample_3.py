import torch
import numpy as np
import torch.utils.data as utils

import numpy as np
import os
from tqdm import tqdm


rootpath = r'PATH TO TRAINING DATASET'
dataset = np.load(os.path.join(rootpath, 'data.npy'), allow_pickle=True) # completed monitoring time series data
dt = np.load(os.path.join(rootpath, 'dt.npy'), allow_pickle=True) # time interval between input vectors
y = np.load(os.path.join(rootpath, 'y.npy'), allow_pickle=True) # label

class GRUD_ODECell(torch.nn.Module):
  def __init__(self, input_size=33, hidden_size=10):
    """
    param:
      input_size (int):  number of features, default 33
      hidden_size (int):  length of hidden vector, default 10
    
    return:
      None
    """
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


  def forward(self, h, x, m, d, prex, mean):
    """
    param:
      h (tensor): hidden vector tensor
      x (tensor): current input vector tensor
      m (tensor): mask of  input vector to indicate if  value is real collected
      d (tensor): time interval sequence
      prex (tensor): last vector tensor
      mean (tensor):  average value of  features

    return:
      dh (tensor):  ODE of hidden vector
      x (tensor): current processed input vector
    """
    gamma_x = torch.exp(-torch.max(self.inputzeros, (self.w_dg_x*d + self.b_dg_x)))
    x = m * x + (1 - m) * (gamma_x * prex + (1 - gamma_x) * mean)

    r = torch.sigmoid(self.lin_xr(x) + self.lin_hr(h) + self.lin_mr(m))
    z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h) + self.lin_mz(m))
    u = torch.tanh(self.lin_xh(x) + self.lin_hu(r * h) + self.lin_mu(m))
    dh = z * (u - h)

    return dh, x

class GRUD_ODE(torch.nn.Module):  
  def __init__(self, input_size=33, hidden_size=10, output_size=2):
    """
    param:
      input_size (int):  number of features, default 33
      hidden_size (int):  length of hidden vector, default 10
      output_size (int):  number of predicted result, default 2

    return:
      None
    """
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.cell = GRUD_ODECell(input_size, hidden_size)
    self.lin = torch.nn.Linear(hidden_size, output_size, bias=True)
    self.reset_parameters()
  
  def reset_parameters(self):
    """
    """
    for params in self.parameters():
      torch.nn.init.normal_(params, mean=0, std=0.1)

  def forward(self, input, dt):
    """
    param:
      input (tensor):  series of input vector
      dt (int): time interval sequence
      
    return:
      None
    """
    X = torch.squeeze(input[0]) 
    Mask = torch.squeeze(input[1]) 
    Delta = torch.squeeze(input[2]) 
    #dt = torch.squeeze(dt) 
    h = torch.autograd.Variable(torch.zeros(self.hidden_size))
    prex = torch.autograd.Variable(torch.zeros(self.input_size))
    mean = torch.squeeze(torch.sum(X,1))/(1e-6+torch.squeeze(torch.sum((Mask!=0),1)))
    for layer in range(X.shape[1]):
      x = torch.squeeze(X[:,layer])
      m = torch.squeeze(Mask[:,layer])
      d = torch.squeeze(Delta[:,layer])
      dh, prex = self.cell(h, x, m, d, prex, mean)
      h = h + dt[layer]*dh
      
    output = self.lin(h)      
    output = torch.sigmoid(output)
    return output


def train_model(model, criterion, learning_rate,\
        train_dataloader,\
        learning_rate_decay, n_epochs=50):
  """
    param:
      model (torch.nn.Module): system model
      criterion (torch.nn.Module): criterion model
      learning_rate (float): learning rate of training phase, default 0.01
      train_dataloader (torch.utils.data.DataLoader):  data loader of training data
      learning_rate_decay (float):  decay value of learning rate
      n_epochs (int): number of times to traverse  training set, default 50
      
    return:
      model(torch.nn.Module): trained system model
  """
  for epoch in range(n_epochs):   
    if learning_rate_decay != 0:
      if  epoch % learning_rate_decay == 0:
        learning_rate = learning_rate/2
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif learning_rate_decay == 0:
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
      
    # train  model
    losses = []
    model.train()
    for train_data, train_label, dt in tqdm(train_dataloader):
      optimizer.zero_grad() 
      train_data = torch.squeeze(train_data)
      train_label = torch.squeeze(train_label)
      dt = torch.squeeze(dt) 
      y_pred = model(train_data, dt)
      loss = criterion(y_pred, train_label)
      losses.append(loss.item())

      loss.backward()
      optimizer.step()
  return model

def data_dataloader(dataset, outcomes, dt):
  """
    param:
      dataset (numpy.ndarray):  data set of training data
      outcomes (numpy.ndarray): label of training data
      dt (numpy.ndarray): time interval sequence of training data
      
    return:
      train_dataloader (torch.utils.data.DataLoader):  data loader of training data
  """
  # ndarray to tensor
  train_data, train_label = torch.Tensor(dataset), torch.Tensor(outcomes)
  train_dt = torch.Tensor(dt)
  
  # tensor to dataset
  train_dataset = utils.TensorDataset(train_data, train_label, train_dt)
  # dataset to dataloader 
  train_dataloader = utils.DataLoader(train_dataset)

  
  return train_dataloader

def train_entrance(): 
  input_size = 33 
  hidden_size = 10 
  output_size = 4
  
  model = GRUD_ODE(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout=0.0)
  criterion = torch.nn.BCELoss()
  
  learning_rate = 0.01
  learning_rate_decay = 5
  n_epochs = 100
  train_dataloader = data_dataloader(dataset, y, dt)
  train_model(model, criterion, learning_rate,\
      train_dataloader,\
      learning_rate_decay, n_epochs)


def inference(model_pretrained, data, dt):
  """
    param:
      model_pretrained (torch.nn.Module):  pre-trained system model
      data (numpy.ndarray): input vector sequence
      dt (numpy.ndarray): time interval sequence of input vector sequence
      
    return:
      mortality (float):  predicted mortality
      length_of_stay (float):  predicted probability of a hospital stay longer than three days
  """
  data_tensor = torch.tensor(data)
  dt_tensor = torch.tensor(dt)
  y_pred = model_pretrained(data_tensor, dt_tensor)
  mortality, length_of_stay = float(y_pred[0]), float(y_pred[1])
  return mortality, length_of_stay