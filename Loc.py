import torch
import torch.nn as nn
import torch.nn.functional as F


class Road(nn.Module):
    def __init__(self):
        super(Road, self).__init__()
        self.build()

    def build(self):
        self.embedding = nn.Embedding(256*256, 16)
        self.process_coords = nn.Linear(2+32, 32)
        
        for module in self.modules():
            if type(module) is not nn.Embedding:
                continue
            nn.init.uniform_(module.state_dict()['weight'], a=-1, b=1)

    def forward(self, traj):
        # road network structure layer
        lngs = torch.unsqueeze(traj['lngs'], dim = 2)
        lats = torch.unsqueeze(traj['lats'], dim = 2)
        grid_ids = torch.unsqueeze(traj['grid_id'].long(), dim = 2)
        grids = torch.squeeze(self.embedding(grid_ids))
        roads = traj['SDNE_embedding']
        
        locs = torch.cat([lngs, lats, grids, roads], dim = 2)
        locs = self.process_coords(locs)
        locs = F.tanh(locs)
        
        return locs

