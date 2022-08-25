import torch


from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

class PhysoinetDatasset(Dataset):
    def __init__(self, list_df, data_root, label_index) -> None:
        super().__init__()
        self.list_df = list_df
        self.data_root = data_root
        self.label_indx = label_index
    
    def __len__(self):
        return len(self.list_df)
    

    def __getitem__(self, index):
        stay = self.list_df['stay'][index]
        series = np.load(os.path.join(self.data_root, 'timeseries', stay.replace('.csv', '.npy')))
        if len(series.shape) != 2 or len(series) <= 0:
            print(stay)
        mask = np.load(os.path.join(self.data_root, 'mask', stay.replace('.csv', '.npy')))
        delta = np.load(os.path.join(self.data_root, 'delta', stay.replace('.csv', '.npy')))
        dt = np.load(os.path.join(self.data_root, 'dt', stay.replace('.csv', '.npy')))
        y = np.array(self.list_df.iloc[index, 2:], dtype=np.uint8)
        y = y[self.label_indx]
        series = torch.tensor(series, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        delta = torch.tensor(delta, dtype=torch.float32)
        dt = torch.tensor(dt, dtype=torch.float32)
        a, b = torch.min(dt), torch.max(dt)
        dt = dt / 12

        y = torch.tensor(y, dtype=torch.float32)

        return series, mask, delta, dt, y
    
class PhysoinetDatassetCache(Dataset):
    def __init__(self, list_df, data_root, label_index, cache_size=1000) -> None:
        super().__init__()
        self.list_df = list_df
        self.data_root = data_root
        self.label_indx = label_index
        self.cache_size = cache_size
        self.label_dict = self.get_label_dict()
        self.cache_buffer = self.cache_data()
        
    def get_label_dict(self):
        label_dict = {}
        for idx, row in self.list_df.iterrows():
            label_dict[row['stay']] = np.array(row[2:].tolist(), dtype=np.uint8)[self.label_indx]
        return label_dict

    def __len__(self):
        return len(self.list_df)
    
    def cache_data(self):
        cache_buffer = {}
        stay_list = self.list_df['stay'].tolist()
        for stay in tqdm(stay_list):
            series = np.load(os.path.join(self.data_root, 'timeseries', stay.replace('.csv', '.npy')))
            if len(series.shape) != 2 or len(series) <= 0:
                print(stay)
            mask = np.load(os.path.join(self.data_root, 'mask', stay.replace('.csv', '.npy')))
            delta = np.load(os.path.join(self.data_root, 'delta', stay.replace('.csv', '.npy')))
            dt = np.load(os.path.join(self.data_root, 'dt', stay.replace('.csv', '.npy')))
            y = self.label_dict[stay]
            cache_buffer[stay] = [series, mask, delta, dt, y]
        return cache_buffer

    def __getitem__(self, index):
        stay = self.list_df['stay'][index]
        series, mask, delta, dt, y = self.cache_buffer[stay]

        series = torch.tensor(series, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        delta = torch.tensor(delta, dtype=torch.float32)
        dt = torch.tensor(dt, dtype=torch.float32)
        dt = dt / 12
        y = torch.tensor(y, dtype=torch.float32)

        return series, mask, delta, dt, y
    
if __name__ == '__main__':
    ds = PhysoinetDatasset(pd.read_csv(r'/media/liu/Data/DataSet/MIMIC/Phenotyping/Sampled100/test_listfile.csv'), r'/media/liu/Data/DataSet/MIMIC/Phenotyping/Sampled100/test')
    dl = DataLoader(ds)
    for  series, mask, delta, dt, y in dl:
        continue