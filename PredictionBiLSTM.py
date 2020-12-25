import copy
import torch
import torch.nn as nn
from torch.autograd import Variable

from Attr import Attr
from SpeedRoadLSTM import SpeedLSTM, RoadLSTM

class PredictionBiLSTM(nn.Module):

    def __init__(self):
        super(PredictionBiLSTM, self).__init__()
        self.build()
    
    def build(self):
        self.attr_net = Attr()
        self.speed_lstm = SpeedLSTM()
        self.road_lstm = RoadLSTM()
        self.bi_lstm = nn.LSTM(
            input_size = self.attr_net.out_size() + 64, \
            hidden_size = 64, \
            num_layers = 2, \
            batch_first = True, \
            bidirectional = True, \
            dropout = 0.25
        )
        
        self.lnhiddens = nn.LayerNorm(self.attr_net.out_size() + 64, elementwise_affine=True)
#        nn.init.uniform_(self.bi_lstm.state_dict()['weight_hh_l0'], a=-0.05, b=0.05)
        
    def forward(self, attr, traj):
        speeds_t = self.speed_lstm(attr, traj)
        roads_t = self.road_lstm(attr, traj)
        
        attr_t = self.attr_net(attr)
        attr_t = torch.unsqueeze(attr_t, dim = 1)
        expand_attr_t = attr_t.expand(roads_t.size()[:2] + (attr_t.size()[-1], ))

        hiddens = torch.cat([expand_attr_t, speeds_t, roads_t], dim = 2)
        hiddens = self.lnhiddens(hiddens)
        lens = copy.deepcopy(traj['lens'])
        lens = list(map(lambda x: x, lens))
        
        packed_inputs = nn.utils.rnn.pack_padded_sequence(hiddens, lens, batch_first = True)
        packed_hiddens, (h_n, c_n) = self.bi_lstm(packed_inputs)
        hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first = True)

        return hiddens
        