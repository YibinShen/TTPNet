import time
import utils

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import ujson as json

class MySet(Dataset):
    def __init__(self, input_file):
        
        DATA_PATH = '../../../dataset/Shanghai/ShanghaiTaxi_iTCL/'
        READ_DATA_PATH = DATA_PATH+'ShanghaiTaxi_iTCL_Train_Whole/'
        self.content = open(READ_DATA_PATH + input_file, 'r').readlines()
        self.content = list(map(lambda x: json.loads(x), self.content))
        self.lengths = list(map(lambda x: len(x['lngs']), self.content))

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(list(self.content))

def collate_fn(data):
    stat_attrs = ['dist', 'time']
    info_attrs = ['driverID', 'dateID', 'weekID', 'timeID']

    traj_attrs = ['lngs', 'lats', 'grid_id', 'time_gap', 'grid_len',
                  'speeds_forward', 'speeds_history', 
                  'speeds_adjacent1', 'speeds_adjacent2',
                  'SDNE_embedding']

    attr, traj = {}, {}

    lens = np.asarray([len(item['lngs']) for item in data])

    for key in stat_attrs:
        x = torch.FloatTensor([item[key] for item in data])
        attr[key] = utils.normalize(x, key)

    for key in info_attrs:
        attr[key] = torch.LongTensor([item[key] for item in data])
    
    for key in traj_attrs:
        if key == 'SDNE_embedding':
            x = np.asarray([item[key] for item in data])
            mask_deepwalk = np.arange(lens.max()*16) < lens[:, None]*16
            padded = np.zeros(mask_deepwalk.shape, dtype = np.float32)
            padded[mask_deepwalk] = np.concatenate(x)
            
            padded = torch.from_numpy(padded).float()
            padded = padded.reshape(padded.shape[0], -1, 16)
            traj[key] = padded
                
        elif (key == 'speeds_forward') or (key == 'speeds_adjacent1') or (key == 'speeds_adjacent2'):
            x = np.asarray([item[key] for item in data])
            mask_speeds_forward = np.arange(lens.max()*4) < lens[:, None]*4
            padded = np.zeros(mask_speeds_forward.shape, dtype = np.float32)
            padded[mask_speeds_forward] = np.concatenate(x)
            
#            padded = utils.normalize(padded, key)
            
            padded = torch.from_numpy(padded).float()
            padded = padded.reshape(padded.shape[0], -1, 4)
            traj[key] = padded
        
        elif key == 'speeds_history':
            x = np.asarray([item[key] for item in data])
            mask_speeds_history = np.arange(lens.max()*7) < lens[:, None]*7
            padded = np.zeros(mask_speeds_history.shape, dtype = np.float32)
            padded[mask_speeds_history] = np.concatenate(x)
            
#            padded = utils.normalize(padded, key)
            
            padded = torch.from_numpy(padded).float()
            padded = padded.reshape(padded.shape[0], -1, 7)
            traj[key] = padded
            
        elif key == 'grid_id':
            x = np.asarray([item[key] for item in data])
            mask = np.arange(lens.max()) < lens[:, None]
            padded = np.zeros(mask.shape, dtype = np.float32)
            padded[mask] = np.concatenate(x)
            
            padded = torch.LongTensor(padded)
            traj[key] = padded

        elif key == 'time_gap':
            x = np.asarray([item[key] for item in data])
#            time = np.asarray([item['time'] for item in data])
            mask = np.arange(lens.max()) < lens[:, None]
            padded = np.ones(mask.shape, dtype = np.float32)
            padded[mask] = np.concatenate(x)
            
            # label
            T_f = torch.from_numpy(padded).float()
            T_f = T_f[:, 1:]
#            time = torch.from_numpy(time).float()
#            time = torch.unsqueeze(time, dim = 1)
#            T_b = time - T_f
#            T_b = T_b[:, :-1]
#            T_b[T_b == 0] = 1.0
            # dual_loss
            mask_f = mask[:, 1:]
#            mask_b = mask[:, 2:]
            M_f = np.zeros(mask_f.shape, dtype = np.int)
#            M_b = np.zeros(mask_b.shape, dtype = np.int)
            M_f[mask_f] = 1
#            M_b[mask_b] = 1
            M_f = torch.from_numpy(M_f).float()
#            M_b = torch.from_numpy(M_b).float()
            
            traj['T_f'] = T_f
#            traj['T_b'] = T_b
            traj['M_f'] = M_f
#            traj['M_b'] = M_b
        
        elif key == 'grid_len':
            x = np.asarray([item[key] for item in data])
            mask = np.arange(lens.max()) < lens[:, None]
            padded = np.zeros(mask.shape, dtype = np.float32)
            padded[mask] = np.concatenate(x)
            
            padded = torch.from_numpy(padded).float()
            traj[key] = padded
            
        else:
            x = np.asarray([item[key] for item in data])
            mask = np.arange(lens.max()) < lens[:, None]
            padded = np.zeros(mask.shape, dtype = np.float32)
            padded[mask] = np.concatenate(x)
            
            padded = utils.normalize(padded, key)
            
            padded = torch.from_numpy(padded).float()
            traj[key] = padded

    lens = lens.tolist()
    traj['lens'] = lens

    return attr, traj


class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        self.lengths = dataset.lengths
        self.indices = list(range(self.count))

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        np.random.shuffle(self.indices)
        chunk_size = self.batch_size * 100
        chunks = (self.count + chunk_size - 1) // chunk_size

        # re-arrange indices to minimize the padding
        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size]
            partial_indices.sort(key = lambda x: self.lengths[x], reverse = True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        # yield batch
        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size

def get_loader(input_file, batch_size):
    dataset = MySet(input_file = input_file)

    batch_sampler = BatchSampler(dataset, batch_size)

    data_loader = DataLoader(dataset = dataset, 
                             batch_size = 1, 
                             collate_fn = lambda x: collate_fn(x), 
                             num_workers = 4,
                             batch_sampler = batch_sampler,
                             pin_memory = True
    )

    return data_loader
