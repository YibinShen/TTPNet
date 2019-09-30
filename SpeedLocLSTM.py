import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from Attr import Attr
from Speed import ShortSpeed, LongSpeed
from Loc import Loc


class SpeedLSTM(nn.Module):

    def __init__(self, ):
        super(SpeedLSTM, self).__init__()
#        self.attr_net = Attr()
        self.shortspeed_net = ShortSpeed()
        self.longspeed_net = LongSpeed()
        self.process_speeds = nn.Linear(64, 32)
#        self.process_speeds_hiddens = nn.Linear(64, 32)
        self.speed_lstm = nn.LSTM(
            input_size = 32, \
            hidden_size = 32, \
            num_layers = 1, \
            batch_first = True, \
            bidirectional = False, \
            dropout = 0
        )

        nn.init.uniform_(self.speed_lstm.state_dict()['weight_hh_l0'], a=-0.05, b=0.05)

    def forward(self, attr, traj):
        shortspeeds_t = self.shortspeed_net(traj)
        longspeeds_t = self.longspeed_net(traj)
#        attr_t = self.attr_net(attr)
#        attr_t = torch.unsqueeze(attr_t, dim = 1)
#        expand_attr_t = attr_t.expand(speeds_t.size()[:2] + (attr_t.size()[-1], ))
#        whole_t = torch.cat([expand_attr_t, speeds_t], dim = 2)
        whole_t = torch.cat([shortspeeds_t, longspeeds_t], dim = 2)
        whole_t = self.process_speeds(whole_t)
        whole_t = F.tanh(whole_t)
        
        lens = copy.deepcopy(traj['lens'])
        lens = list(map(lambda x: x, lens))
        
        packed_inputs = nn.utils.rnn.pack_padded_sequence(whole_t, lens, batch_first = True)
        packed_hiddens, (h_n, c_n) = self.speed_lstm(packed_inputs)
        speeds_hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first = True)
        
#        speeds_hiddens = self.process_speeds_hiddens(speeds_hiddens)
#        speeds_hiddens = F.tanh(speeds_hiddens)
        
        return speeds_hiddens


class LocLSTM(nn.Module):

    def __init__(self, ):
        super(LocLSTM, self).__init__()
#        self.attr_net = Attr()
        self.loc_net = Loc()
#        self.process_locs_hiddens = nn.Linear(64, 32)
        self.loc_lstm = nn.LSTM(
            input_size = 32, \
            hidden_size = 32, \
            num_layers = 1, \
            batch_first = True, \
            bidirectional = False, \
            dropout = 0
        )

        nn.init.uniform_(self.loc_lstm.state_dict()['weight_hh_l0'], a=-0.05, b=0.05)

    def forward(self, attr, traj):
        locs_t = self.loc_net(traj)
#        attr_t = self.attr_net(attr)
#        attr_t = torch.unsqueeze(attr_t, dim = 1)
#        expand_attr_t = attr_t.expand(locs_t.size()[:2] + (attr_t.size()[-1], ))
#        whole_t = torch.cat([expand_attr_t, locs_t], dim = 2)
        whole_t = locs_t
        
        lens = copy.deepcopy(traj['lens'])
        lens = list(map(lambda x: x, lens))
        
        packed_inputs = nn.utils.rnn.pack_padded_sequence(whole_t, lens, batch_first = True)
        packed_hiddens, (h_n, c_n) = self.loc_lstm(packed_inputs)
        locs_hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first = True)
        
#        locs_hiddens = self.process_locs_hiddens(locs_hiddens)
#        locs_hiddens = F.tanh(locs_hiddens)
        
        return locs_hiddens
