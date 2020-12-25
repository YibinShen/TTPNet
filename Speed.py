import utils
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortSpeed(nn.Module):
    def __init__(self):
        super(ShortSpeed, self).__init__()
        self.build()
    
    def build(self):
#        self.process_shortspeeds = nn.Linear(48, 16)
        self.short_kernel_size = 2
        self.short_cnn = nn.Conv1d(3, 4, kernel_size = self.short_kernel_size, stride = 1)
        self.short_rnn = nn.RNN(
            input_size = 4, \
            hidden_size = 16, \
            num_layers = 1, \
            batch_first = True
        )
        
#        nn.init.uniform_(self.short_rnn.state_dict()['weight_hh_l0'], a=-0.05, b=0.05)
    
    def forward(self, traj):
        # short-term travel speed features
        n_batchs = traj['speeds_0'].size()[0]
        speeds_forward = traj['speeds_0'].reshape(-1, 4)
        speeds_adjacent1 = traj['speeds_1'].reshape(-1, 4)
        speeds_adjacent2 = traj['speeds_2'].reshape(-1, 4)
        grid_len = traj['grid_len'].reshape(-1, 1)
        
        speeds_forward = torch.unsqueeze(speeds_forward, dim =2)
        speeds_adjacent1 = torch.unsqueeze(speeds_adjacent1, dim = 2)
        speeds_adjacent2 = torch.unsqueeze(speeds_adjacent2, dim = 2)

        grid_len = torch.unsqueeze(grid_len, dim = 2)
        grid_len_short = grid_len.expand(speeds_forward.size()[:2] + (grid_len.size()[-1], ))
        
        times_forward = speeds_forward.clone()
        times_forward[times_forward==0] = 0.2
        times_forward = grid_len_short / times_forward * 3600
        times_adjacent1 = speeds_adjacent1.clone()
        times_adjacent1[times_adjacent1==0] = 0.2
        times_adjacent1 = grid_len_short / times_adjacent1 * 3600
        times_adjacent2 = speeds_adjacent2.clone()
        times_adjacent2[times_adjacent2==0] = 0.2
        times_adjacent2 = grid_len_short / times_adjacent2 * 3600
        
        speeds_forward = utils.normalize(speeds_forward, 'speeds_0')
        speeds_adjacent1 = utils.normalize(speeds_adjacent1, 'speeds_1')
        speeds_adjacent2 = utils.normalize(speeds_adjacent2, 'speeds_2')
        grid_len_short = utils.normalize(grid_len_short, 'grid_len')
        times_forward = utils.normalize(times_forward, 'time_gap')
        times_adjacent1 = utils.normalize(times_adjacent1, 'time_gap')
        times_adjacent2 = utils.normalize(times_adjacent2, 'time_gap')
        
        inputs_0 = torch.cat([speeds_forward, grid_len_short, times_forward], dim = 2)
        inputs_1 = torch.cat([speeds_adjacent1, grid_len_short, times_adjacent1], dim = 2)
        inputs_2 = torch.cat([speeds_adjacent2, grid_len_short, times_adjacent2], dim = 2)
        
        outputs_0 = F.tanh(self.short_cnn(inputs_0.permute(0, 2, 1)))
        outputs_0 = outputs_0.permute(0, 2, 1)
        outputs_1 = F.tanh(self.short_cnn(inputs_1.permute(0, 2, 1)))
        outputs_1 = outputs_1.permute(0, 2, 1)
        outputs_2 = F.tanh(self.short_cnn(inputs_2.permute(0, 2, 1)))
        outputs_2 = outputs_2.permute(0, 2, 1)

        outputs_0, h_n = self.short_rnn(outputs_0)
        outputs_1, h_n = self.short_rnn(outputs_1)
        outputs_2, h_n = self.short_rnn(outputs_2)

        outputs_0 = outputs_0.reshape(n_batchs, -1, 4-self.short_kernel_size+1, 16)
        outputs_1 = outputs_1.reshape(n_batchs, -1, 4-self.short_kernel_size+1, 16)
        outputs_2 = outputs_2.reshape(n_batchs, -1, 4-self.short_kernel_size+1, 16)
        
        V_short = torch.cat([outputs_0[:, :, -1], outputs_1[:, :, -1], outputs_2[:, :, -1]], dim = 2)
#        V_short = self.process_shortspeeds(V_short)
#        V_short = F.tanh(V_short)
        
        return V_short


class LongSpeed(nn.Module):
    def __init__(self):
        super(LongSpeed, self).__init__()
        self.build()
    
    def build(self):
#        self.process_longspeeds = nn.Linear(16, 16)
        self.long_kernel_size = 3
        self.long_cnn = nn.Conv1d(3, 4, kernel_size = self.long_kernel_size, stride = 1)
        self.long_rnn = nn.RNN(
            input_size = 4, \
            hidden_size = 16, \
            num_layers = 1, \
            batch_first = True
        )

#        nn.init.uniform_(self.long_rnn.state_dict()['weight_hh_l0'], a=-0.05, b=0.05)
    
    def forward(self, traj):
        # long-term travel speed features
        n_batchs = traj['speeds_long'].size()[0]
        speeds_history = traj['speeds_long'].reshape(-1, 7)
        grid_len = traj['grid_len'].reshape(-1, 1)
        
        speeds_history = torch.unsqueeze(speeds_history, dim = 2)

        grid_len = torch.unsqueeze(grid_len, dim = 2)
        grid_len_long = grid_len.expand(speeds_history.size()[:2] + (grid_len.size()[-1], ))
        
        times_history = speeds_history.clone()
        times_history[times_history==0] = 0.2
        times_history = grid_len_long / times_history * 3600
        
        speeds_history = utils.normalize(speeds_history, 'speeds_long')
        grid_len_long = utils.normalize(grid_len_long, 'grid_len')
        times_history = utils.normalize(times_history, 'time_gap')
        
        inputs_3 = torch.cat([speeds_history, grid_len_long, times_history], dim = 2)
        outputs_3 = self.long_cnn(inputs_3.permute(0, 2, 1))
        outputs_3 = outputs_3.permute(0, 2, 1)
        outputs_3, h_n = self.long_rnn(outputs_3)
        outputs_3 = outputs_3.reshape(n_batchs, -1, 7-self.long_kernel_size+1, 16)
        
        V_long = outputs_3[:, :, -1]
#        V_long = self.process_longspeeds(V_long)
#        V_long = F.tanh(V_long)
        
        return V_long