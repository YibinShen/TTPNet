import torch
from torch.autograd import Variable

import json

from math import radians, cos, sin, asin, sqrt

config = json.load(open('Config/Config_128.json', 'r'))

def geo_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, map(float, [lon1, lat1, lon2, lat2]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def normalize(x, key):
    mean = config[key + '_mean']
    std = config[key + '_std']
    return (x - mean) / std

def unnormalize(x, key):
    mean = config[key + '_mean']
    std = config[key + '_std']
    return x * std + mean

def pad_sequence(sequences, lengths):
    padded = torch.zeros(len(sequences), lengths[0]).float()
    for i, seq in enumerate(sequences):
        seq = torch.Tensor(seq)
        padded[i, :lengths[i]] = seq[:]
    return padded

def to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x), var)
        return var
