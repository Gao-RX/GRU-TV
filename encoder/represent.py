import os
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from cells import GRUTCell
from encoder_decoder import GRUAutoEncoder
from dataset import data_dataloader

from config_ae import BaseConfig
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns

my_config = BaseConfig()
my_config.set_config('gamma_x', False)
my_config.set_config('gamma_h', False)
my_config.set_config('load_old', True)
my_config.set_config('desc', 'GRUT-AE-hid=64')

config = my_config.get_config()
curr_dir = os.path.dirname(__file__)
result_dir = os.path.join(curr_dir, 'result', config['desc']+'_'+config['sample'])
os.makedirs(result_dir, exist_ok=True)
my_config.record_config(os.path.join(result_dir, 'config.txt'))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def represent(model: nn.Module, dataloader: DataLoader):
    model.eval()
    rep_list = []
    label_list = []
    for train_data, train_label, dt in tqdm(dataloader):
        train_data = torch.transpose(train_data, 2, 3)
        sequence = train_data[:, 0, ...].cuda()
        mask = train_data[:, 1, ...].cuda()
        delta = train_data[:, 2, ...].cuda()
        dt = dt.cuda()
        represent = model(sequence, mask, delta, dt)
        represent = represent.cpu().data.numpy()
        rep_list.append(represent)
        label_list.append(train_label.numpy())
    np.save(os.path.join(result_dir, 'label.npy'), np.squeeze(np.array(label_list)))
    np.save(os.path.join(result_dir, 'rep_features.npy'), np.squeeze(np.array(rep_list)))

def main():
    dataset = np.load(os.path.join(config['data_dir'], 'dataset_'+config['sample']+'.npy'), allow_pickle=True)
    seq = np.transpose(dataset[:, 0, ...], (1, 0, 2)).reshape((33, -1))
    mean_value = np.ma.masked_less_equal(seq, 0).mean(axis=1)
    mean_value = mean_value.reshape((1, mean_value.shape[0], 1))
    dataset[:, 0, ...] = dataset[:, 0, ...]/mean_value
    dt = np.load(os.path.join(config['data_dir'], 'dt_'+config['sample']+'.npy'), allow_pickle=True)
    dt[:, 0] = 1
    y = np.load(os.path.join(config['data_dir'], 'y.npy'), allow_pickle=True)
    train_dataloader, dev_dataloader, test_dataloader = data_dataloader(dataset, y, dt)
    encoder = GRUTCell(input_size=config['input_size'], hidden_size=config['hidden_size'], gamma_x=config['gamma_x'], gamma_h=config['gamma_h'])
    decoder = GRUTCell(input_size=config['input_size'], hidden_size=config['hidden_size'], gamma_x=config['gamma_x'], gamma_h=config['gamma_h'])
    ae = GRUAutoEncoder(encoder, decoder, input_size=config['input_size'], hidden_size=config['hidden_size'], feature_size=config['feature_size'], is_rep=True)
    ckpt = torch.load(os.path.join(result_dir, 'best.ckpt'))
    ae.load_state_dict(ckpt['model_state_dict'])
    ae = ae.cuda()
    represent(ae, test_dataloader)

def t_sne_visu():
    save_dir = r'/home/liu/Desktop/0_features'
    features = np.load(os.path.join(result_dir, 'rep_features.npy'))
    labels = np.load(os.path.join(result_dir, 'label.npy')).tolist()
    x_embedded = TSNE(n_components=2, init='random').fit_transform(features)
    label_bin = list(map(lambda x: 0 if x[0]==0 else 1, labels))
    df = pd.DataFrame({'X': x_embedded[:, 0].tolist(), 'Y': x_embedded[:, 1].tolist(), 'Label': label_bin})
    sns.scatterplot(data=df, x='X', y='Y', hue='Label')
    plt.savefig(os.path.join(save_dir, str(len(os.listdir(save_dir))+1)+'.png'))
    plt.close()

def k_means_visu():
    features = np.load(os.path.join(result_dir, 'rep_features.npy'))
    features = np.nan_to_num(features)
    kmeans = KMeans(n_clusters=2, random_state=0, init='k-means++').fit(features)
    centroid_values = kmeans.cluster_centers_
    labels = np.load(os.path.join(result_dir, 'label.npy')).tolist()
    label_bin = list(map(lambda x: 0 if x[2]==0 else 1, labels))
    plt.scatter(features[:, 0], features[:, 1], c=label_bin, s=50, alpha=0.5)
    plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=35)
    # df = pd.DataFrame({'X': features[:, 0].tolist(), 'Y': features[:, 1].tolist(), 'Label': label_bin})
    # sns.scatterplot(data=df, x='X', y='Y', hue='Label', alpha=0.2)
    plt.show()


if __name__ == '__main__':
    # main()
    for i in range(100):
        t_sne_visu()
    # k_means_visu()
