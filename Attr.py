import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import numpy as np

from torch.autograd import Variable

class Attr(nn.Module):
    embed_dims = [('driverID', 13000, 8), ('weekID', 7, 3), ('timeID', 96, 8)]

    def __init__(self):
        super(Attr, self).__init__()
        # whether to add the two ends of the path into Attribute Component
        self.build()

    def build(self):
        for name, dim_in, dim_out in Attr.embed_dims:
            self.add_module(name + '_em', nn.Embedding(dim_in, dim_out))
        
#        for module in self.modules():
#            if type(module) is not nn.Embedding:
#                continue
#            nn.init.uniform_(module.state_dict()['weight'], a=-1, b=1)

    def out_size(self):
        sz = 0
        for name, dim_in, dim_out in Attr.embed_dims:
            sz += dim_out
        
        return sz + 2

    def forward(self, attr):
        em_list = []
        for name, dim_in, dim_out in Attr.embed_dims:
            embed = getattr(self, name + '_em')
            attr_t = attr[name].view(-1, 1)

            attr_t = torch.squeeze(embed(attr_t))

            em_list.append(attr_t)

        dist = attr['dist']
        em_list.append(dist.view(-1, 1))
        em_list.append(attr['dateID'].float().view(-1, 1))

        return torch.cat(em_list, dim = 1)
