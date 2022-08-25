import numpy as np
import torch
import torch.utils.data as utils

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
