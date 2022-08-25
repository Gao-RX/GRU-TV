import datetime

class BaseConfig():
    def __init__(self):
        self.config = {
            # train config
            'epoch_max': 200,
            'break_epoch': 3,
            'input_size': 33,
            'hidden_size': 128,
            'feature_size': 2,
            'is_cuda': True,
            'print_freq': 1,
            'val_freq': 1,
            'init_lr': 0.003,
            'lr_decay_freq': 3,
            'is_shuffle': True,
            'num_works': 0,
            'num_class': 1,
            'seed': 42,

            # path config
            'data_dir': r'/media/liu/Data/DataSet/Physionet2012/Sampled',
            'sample': '50',

            # task description
            'desc': 'GRUT-AE-hid=2',
            'trick': ''
        }
    
    def set_config(self, key, value):
        self.config[key] = value
    
    def record_config(self, path):
        file = open(path, mode='a+', encoding='utf8')
        desc_str = self.__str__()
        file.write(desc_str)
    
    def get_config(self):
        return self.config

    def __str__(self):
        desc_string = ''
        for key in self.config.keys():
            desc_string += '{}: {}\n'.format(key, str(self.config[key]))
        desc_string += '*'*50+str(datetime.datetime.now()) + '*'*50+'\n'
        return desc_string
