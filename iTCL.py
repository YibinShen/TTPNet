import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from PredictionBiLSTM import PredictionBiLSTM

class iTCL(nn.Module):
    def __init__(self, ):
        super(iTCL, self).__init__()

        self.build()
        self.init_weight()

    def build(self):
        self.bi_lstm = PredictionBiLSTM()
        
        self.input2hid = nn.Linear(128, 128)
        self.hid2hid = nn.Linear(128, 64)
        self.hid2out = nn.Linear(64, 1)
        
    def forward(self, attr, traj):
        hiddens = self.bi_lstm(attr, traj)
        n = hiddens.size()[1]
        h_f = []
        
        for i in range(2, n):
            h_f_temp = torch.sum(hiddens[:, :i], dim = 1)
            h_f.append(h_f_temp)
            
        h_f.append(torch.sum(hiddens, dim = 1))
        h_f = torch.stack(h_f).permute(1, 0, 2)
        
        T_f_hat = self.input2hid(h_f)
        T_f_hat = F.leaky_relu(T_f_hat)
        T_f_hat = self.hid2hid(T_f_hat)
        T_f_hat = F.leaky_relu(T_f_hat)
        T_f_hat = self.hid2out(T_f_hat)

        return T_f_hat

    def dual_loss(self, T_f_hat, traj, mean, std):
        T_f_hat = T_f_hat * std + mean
        
        T_f = torch.unsqueeze(traj['T_f'], dim = 2)
        M_f = torch.unsqueeze(traj['M_f'], dim = 1)
        
        loss_f = torch.bmm(M_f, torch.pow((T_f_hat-T_f)/T_f, 2)) / torch.bmm(M_f, M_f.permute(0, 2, 1))
        loss_f = torch.pow(loss_f, 1/2)
#        loss_f = torch.bmm(M_f, torch.abs((T_f_hat-T_f)/T_f)) / torch.bmm(M_f, M_f.permute(0, 2, 1))
        
        return {'pred': T_f_hat[:, -1]}, loss_f.mean()
    
    def MAPE_loss(self, pred, label, mean, std):
        label = label.view(-1, 1)
        label = label * std + mean
        
        loss = torch.abs(pred - label) / label
        
        return {'label': label, 'pred': pred}, loss.mean()
    
    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find('.ln') == -1:
                print(name)
                if name.find('.bias') != -1:
                    param.data.fill_(0)
                elif name.find('.weight') != -1:
                    nn.init.xavier_uniform_(param.data)
    
    def eval_on_batch(self, attr, traj, config):
        T_f_hat = self(attr, traj)
        
        pred_dict, loss = self.dual_loss(T_f_hat, traj, 
                                     config['time_gap_mean'], config['time_gap_std'])
        MAPE_dict, MAPE_loss = self.MAPE_loss(pred_dict['pred'], attr['time'], 
                                          config['time_mean'], config['time_std'])
        return pred_dict, loss, MAPE_dict, MAPE_loss
