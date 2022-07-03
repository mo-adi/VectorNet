import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import utils

class MLP(nn.Module):
    '''MLP with layer norm and relu used for the subgraph layers'''
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2) 
        )

    def forward(self, x):
        x = self.MLP(x)
        return x

class MLP2(nn.Module):
    '''MLP for trajectory'''
    def __init__(self, hidden_size, out):
        super(MLP2, self).__init__()
        self.MLP2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out) 
        )

    def forward(self, x):
        x = self.MLP2(x)
        return x


class Global_Graph(nn.Module):
    '''The global graph GNN'''
    def __init__(self, hidden_size):
        super(Global_Graph, self).__init__()
        self.fc = [nn.Linear(hidden_size, hidden_size).cuda() for _ in range(3)]

    def forward(self, inputs, attention_mask, mapping):
        cuda0 = torch.device('cuda:0')
        poly_n, poly_len, hidden_size = inputs.shape
        Q = self.fc[0](inputs)
        K = self.fc[1](inputs)
        V = self.fc[2](inputs)
        out = torch.bmm(Q, K.transpose(1, 2))
        out = torch.bmm(out, attention_mask)
        out = F.softmax(out, dim=2)
        out = torch.bmm(out, V)
        out.squeeze_(1)
        return out


class Sub_Graph(nn.Module):
    '''The entire subgraph, which consists of multiple layers'''
    def __init__(self, hidden_size, depth=3):
        super(Sub_Graph, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.layers = nn.Sequential()
        for i in range(depth):
            self.layers.add_module('sub{}'.format(i), SubLayer(hidden_size))
        
    def forward(self, hidden_states, lengths): # hidden_state.shape = [poly_n, poly_len, hidden_size]
        cuda0 = torch.device('cuda:0')
        poly_n = hidden_states.shape[0]
        max_len = max(lengths)
        mask = torch.zeros([poly_n, max_len, self.hidden_size], device=cuda0)
        for i, length in enumerate(lengths):
            mask[i][:length][:length].fill_(1)
        out = torch.mul(mask, hidden_states)
        out = self.layers(out)
        out = out.permute(0, 2, 1)
        out = F.max_pool1d(out, kernel_size=out.shape[2])
        out = out.permute(0, 2, 1)
        out.squeeze_(1)
        return out


class SubLayer(nn.Module): 
    ''' A layer of the subgraph where the MLP (genc) takes place'''
    def __init__(self, hidden_size): 
        super(SubLayer, self).__init__()
        self.genc = MLP(hidden_size)

    def forward(self, hidden_states):
        assert len(hidden_states.shape) == 3
        hidden_states = self.genc(hidden_states) # MLP
        poly_n, poly_len, hidden_size = hidden_states.shape
        hidden_states2 = hidden_states.permute(0, 2, 1)
        hidden_states2 = F.max_pool1d(hidden_states2, kernel_size=hidden_states2.shape[2])
        hidden_states2 = torch.cat([hidden_states2] * poly_len, dim=2)

        out = torch.cat((hidden_states2.permute(0, 2, 1), hidden_states), dim=2)
        assert out.shape == (poly_n, poly_len, hidden_size*2)
        return out # output should be of shape [poly_n, poly_len, hidden_size*2] because of the concat operation