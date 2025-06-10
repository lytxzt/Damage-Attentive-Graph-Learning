from copy import deepcopy
from torch.optim import Adam
import Utils
from Configurations import *

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.norm import LayerNorm

best_hidden_dimension = 500
best_dropout = 0.1
best_lr=0.01

torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

class GAT:
    def __init__(self):
        self.hidden_dimension = best_hidden_dimension
        self.dropout_value = best_dropout
        self.gcn_network = GAT_fixed_structure(nfeat=config_dimension, nhid=self.hidden_dimension, nclass=config_dimension, dropout=self.dropout_value, if_dropout=True, bias=True)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.gcn_network.cuda()

        self.optimizer = Adam(self.gcn_network.parameters(), lr=best_lr)
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.Integer = torch.cuda.IntTensor if self.use_cuda else torch.IntTensor

    def gat(self, global_positions, remain_list):
        remain_positions = []
        for i in remain_list:
            remain_positions.append(deepcopy(global_positions[i]))
        remain_positions = np.array(remain_positions)
        num_remain = len(remain_list)

        edge_set = []
        for i in range(len(remain_list)):
            for j in range(i+1, len(remain_list)):
                if np.linalg.norm(remain_positions[i]-remain_positions[j]) <= config_communication_range:
                    edge_set.append([i, j])
                    edge_set.append([j, i])

        A_hat = np.array(edge_set).T

        remain_positions = torch.FloatTensor(remain_positions).type(self.FloatTensor)
        A_hat = torch.FloatTensor(A_hat).type(self.Integer)

        best_final_positions = 0
        best_loss = 1000000000000
        loss_ = 0
        counter_loss = 0
        
        for train_step in range(1000):
            if loss_ > 1000 and train_step > 50:
               self.optimizer = Adam(self.gcn_network.parameters(), lr=best_lr)
            if counter_loss > 4 and train_step > 50:
               break

            final_positions = self.gcn_network(remain_positions, A_hat)

            final_positions = 0.5 * torch.Tensor(config_space_range).type(self.FloatTensor) * final_positions

            # check if connected
            final_positions_ = final_positions.cpu().data.numpy()
            A = Utils.make_A_matrix(final_positions_, len(final_positions_), config_communication_range)
            D = Utils.make_D_matrix(A, len(A))
            L = D - A
            flag, num = Utils.check_number_of_clusters(L, len(L))

            # loss
            loss = 1000*(num-1) + torch.max(torch.norm(final_positions-remain_positions, dim=1))
            
            # best loss
            if loss.cpu().data.numpy() < best_loss:
                best_loss = deepcopy(loss.cpu().data.numpy())
                best_final_positions = deepcopy(final_positions.cpu().data.numpy())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_ = loss.cpu().data.numpy()
            if loss_ > 1000 and train_step > 50:
                counter_loss += 1
            print("    episode %d, loss %f" % (train_step, loss_), end='\r')
        
        speed = np.zeros((config_num_of_agents, config_dimension))
        remain_positions_numpy = remain_positions.cpu().data.numpy()
        temp_max_distance = 0

        for i in range(num_remain):
            if np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]) > 0:
                speed[remain_list[i]] = (best_final_positions[i] - remain_positions_numpy[i]) / np.linalg.norm(
                    best_final_positions[i] - remain_positions_numpy[i])
            if np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]) > temp_max_distance:
                temp_max_distance = deepcopy(np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]))

        max_time = temp_max_distance / config_constant_speed
        return deepcopy(speed), deepcopy(max_time), deepcopy(best_final_positions)

     
class GAT_fixed_structure(nn.Module):
    def __init__(self, nfeat=3, nhid=500, nclass=3, n_heads=4, dropout=0.5, alpha=0.2, if_dropout=True, bias=True):
        super(GAT_fixed_structure, self).__init__()

        # input
        self.gat1 = GATv2Conv(in_channels=nfeat, out_channels=nhid, heads=n_heads, concat=True, negative_slope=alpha, dropout=dropout, bias=bias, add_self_loops=True)
        # mid-layers
        self.gat2 = GATv2Conv(in_channels=nhid*n_heads, out_channels=nhid, heads=n_heads, concat=True, negative_slope=alpha, dropout=dropout, bias=bias)
        self.gat3 = GATv2Conv(in_channels=nhid*n_heads, out_channels=nhid, heads=n_heads, concat=True, negative_slope=alpha, dropout=dropout, bias=bias)
        self.gat4 = GATv2Conv(in_channels=nhid*n_heads, out_channels=nhid, heads=n_heads, concat=True, negative_slope=alpha, dropout=dropout, bias=bias)
        self.gat5 = GATv2Conv(in_channels=nhid*n_heads, out_channels=nhid, heads=n_heads, concat=True, negative_slope=alpha, dropout=dropout, bias=bias)
        # output: non-concat
        self.gat6 = GATv2Conv(in_channels=nhid*n_heads, out_channels=nclass, heads=n_heads, concat=False, negative_slope=alpha, dropout=0, bias=bias)
                
        # torch_geometric.nn.norm.LayerNorm
        self.norm1 = LayerNorm(nhid * n_heads)
        self.norm2 = LayerNorm(nhid * n_heads)
        self.norm3 = LayerNorm(nhid * n_heads)
        self.norm4 = LayerNorm(nhid * n_heads)
        self.norm5 = LayerNorm(nhid * n_heads)
        self.norm6 = LayerNorm(nclass)

        self.dropout = dropout
        self.training = if_dropout
        self.bias = bias
    
    def forward(self, x, adj):
        x = self.norm1(self.gat1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.norm2(self.gat2(x, adj))
        x = self.norm3(self.gat3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.norm4(self.gat4(x, adj))
        x = self.norm5(self.gat5(x, adj))

        x = self.norm6(self.gat6(x, adj))

        return torch.tanh(x)+1
