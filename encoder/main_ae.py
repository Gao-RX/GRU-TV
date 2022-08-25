import sys
import torch
from torch import optim
import numpy as np
import os
import torch.nn as nn

from cells import GRUTCell
from encoder_decoder import GRUAutoEncoder
from dataset import data_dataloader
from tqdm import tqdm

from config_ae import BaseConfig

my_config = BaseConfig()
my_config.set_config('gamma_x', False)
my_config.set_config('gamma_h', False)
my_config.set_config('load_old', False)
my_config.set_config('desc', 'GRUT-AE-hid=128')


config = my_config.get_config()
curr_dir = os.path.dirname(__file__)
result_dir = os.path.join(curr_dir, 'result', config['desc']+'_'+config['sample'])
os.makedirs(result_dir, exist_ok=True)
my_config.record_config(os.path.join(result_dir, 'config.txt'))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(model: nn.Module, train_dataloader, test_dataloader, criterion, optimizer, ckpt=None):
    min_loss_test = sys.maxsize
    if ckpt:
        begin_epoch = int(ckpt['epoch'])
    else:
        begin_epoch = 0
    for epoch in range(begin_epoch, config['epoch_max']):
        loss_sum = 0
        optimizer = exp_lr_scheduler(optimizer, epoch)
        for train_data, _, dt in tqdm(train_dataloader):
            train_data = torch.transpose(train_data, 2, 3)
            sequence = train_data[:, 0, ...].cuda()
            mask = train_data[:, 1, ...].cuda()
            delta = train_data[:, 2, ...].cuda()
            dt = dt.cuda()
            optimizer.zero_grad()
            output, length = model(sequence, mask, delta, dt)
            loss = criterion(torch.squeeze(sequence[:, 0:length, :]), torch.squeeze(output))
            loss.backward()
            optimizer.step()
            loss_value = loss.cpu().data.numpy()
            # print(loss_value, end=' ')
            loss_sum += loss_value
        loss = loss_sum / len(train_dataloader)
        print(f'Train, Epoch:{str(epoch)}, Loss:{str(loss)}\n')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'epoch': epoch
            },
            os.path.join(result_dir, 'last.ckpt')
        )

        test_loss = test(model, test_dataloader, criterion)
        if test_loss < min_loss_test:
            min_loss_test = min(test_loss, min_loss_test)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'epoch': epoch
                },
                os.path.join(result_dir, 'best.ckpt')
            )
            print(f'Test, Epoch:{str(epoch)}, Loss:{str(test_loss)}, Saved!\n')
        else:
            print(f'Test, Epoch:{str(epoch)}, Loss:{str(test_loss)}, Pass\n')
def test(model, test_dataloader, criterion):
    model.eval()
    loss_sum = 0
    for train_data, _, dt in tqdm(test_dataloader):
        train_data = torch.transpose(train_data, 2, 3)
        sequence = train_data[:, 0, ...].cuda()
        mask = train_data[:, 1, ...].cuda()
        delta = train_data[:, 2, ...].cuda()
        dt = dt.cuda()
        output, length = model(sequence, mask, delta, dt)
        loss = criterion(torch.squeeze(sequence[:, 0:length, :]), torch.squeeze(output))
        loss_sum += loss.cpu().data.numpy()
    loss = loss_sum / len(test_dataloader)
    model.train()
    return loss


def main():
    dataset = np.load(os.path.join(config['data_dir'], 'dataset_'+config['sample']+'.npy'), allow_pickle=True)
    seq = np.transpose(dataset[:, 0, ...], (1, 0, 2)).reshape((33, -1))
    mean_value = np.ma.masked_less_equal(seq, 0).mean(axis=1)
    min_value = seq.min(axis=1)
    max_value = seq.max(axis=1)
    mean_value = mean_value.reshape((1, mean_value.shape[0], 1))
    dataset[:, 0, ...] = dataset[:, 0, ...]/mean_value
    dt = np.load(os.path.join(config['data_dir'], 'dt_'+config['sample']+'.npy'), allow_pickle=True)
    dt[:, 0] = 1.0
    y = np.load(os.path.join(config['data_dir'], 'y.npy'), allow_pickle=True)
    train_dataloader, dev_dataloader, test_dataloader = data_dataloader(dataset, y, dt)

    encoder = GRUTCell(input_size=config['input_size'], hidden_size=config['hidden_size'], gamma_x=config['gamma_x'], gamma_h=config['gamma_h'])
    decoder = GRUTCell(input_size=config['input_size'], hidden_size=config['hidden_size'], gamma_x=config['gamma_x'], gamma_h=config['gamma_h'])
    ae = GRUAutoEncoder(encoder, decoder, input_size=config['input_size'], hidden_size=config['hidden_size'], feature_size=config['feature_size'])
    if config['load_old']:
        ckpt = torch.load(os.path.join(result_dir, 'last.ckpt'))
        ae.load_state_dict(ckpt['model_state_dict'])
    else:
        ckpt = None
    criterion = nn.MSELoss()
    if config['is_cuda']:
        ae = ae.cuda()
        criterion = criterion.cuda()
    optimizer = optim.Adam(ae.parameters(), lr=config['init_lr'], weight_decay=0.0001)
    if config['load_old']:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])


    train(ae, train_dataloader, test_dataloader, criterion, optimizer, ckpt) # TODO 


def exp_lr_scheduler(optimizer, epoch, init_lr=config['init_lr'], lr_decay_epoch=config['lr_decay_freq'], min_vakue=0.00001):
    lr = max(init_lr * (0.8 ** (epoch // lr_decay_epoch)), min_vakue)
    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer


if __name__ == '__main__':
    main()
