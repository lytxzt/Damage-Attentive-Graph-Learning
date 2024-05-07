from copy import deepcopy
from torch.optim import Adam
import Utils
from Configurations import *
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
# from multiprocessing import Pool
import matplotlib.pyplot as plt

best_hidden_dimension = 512
best_dropout = 0.1
# lr = 0.00001
alpha = 0.12
draw = False
batch_size = 8
embedding_distence = 450

torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

'''
mode 1: tahn    + L1
mode 2: no tahn + L1
mode 3: tahn    + no L1
mode 4: no tahn + no L1
'''

dimension = config_dimension
central_point = np.array([500, 500]) if dimension == 2 else np.array([500, 500, 50])

class DD_GCN:
    def __init__(self):
        self.hidden_dimension = best_hidden_dimension
        self.dropout_value = best_dropout

    def dd_batch(self, global_positions, remain_list):
        # self.gcn_network = GCN_fixed_structure(nfeat=dimension, nhid=self.hidden_dimension, nclass=dimension, dropout=self.dropout_value, if_dropout=True, bias=True)
        gcn_network = GCN_dd_structure(nfeat=dimension, nhid=self.hidden_dimension, nclass=dimension, dropout=self.dropout_value, if_dropout=True, bias=True)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            gcn_network.cuda()

        # self.optimizer = Adam(self.gcn_network.parameters(), lr=0.001)
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        optimizer = Adam(gcn_network.parameters(), lr=0.0001)
        # optimizer = SGD(gcn_network.parameters(), lr=0.01, momentum=0.9)

        remain_positions = []
        for i in remain_list:
            remain_positions.append(deepcopy(global_positions[i]))
        
        num_remain = len(remain_list)
        
        fixed_positions = deepcopy(remain_positions)
        for i in range(len(global_positions)):
            if i not in remain_list:
                fixed_positions.append(deepcopy(global_positions[i]))
        fixed_positions.append(central_point)

        num_of_agents = len(fixed_positions)
        num_destructed = num_of_agents - num_remain
        
        fixed_positions = np.array(fixed_positions)
        remain_positions = np.array(remain_positions)

        # damage differential
        A_hat_batch = []

        all_neighbor = calculate_khop_neighbour(fixed_positions, config_communication_range)
        # batch_size = max([len(hop) for hop in all_neighbor])-1

        for khop in range(2, 2+batch_size):
            A = make_dd_khop_A_matrix(all_neighbor, fixed_positions, num_remain, khop)
            # A = Utils.make_A_matrix(remain_positions, num_remain, config_communication_range)

            D = Utils.make_D_matrix(A, num_of_agents)
            L = D - A
            A_norm = np.linalg.norm(A, ord=np.inf)
            # print(A_norm)
            # A_norm = num_of_agents
            k0 = 1 / A_norm
            K = 0.5 * k0
            A_hat_khop = np.eye(num_of_agents) - K * L

            # D_norm = np.array([num_destructed for _ in range(num_remain)] + [num_remain for _ in range(num_destructed)])
            # D_tilde_sqrt = np.diag(D_norm ** (-0.5))
            # A_hat_khop = np.eye(num_of_agents) - 0.99 * D_tilde_sqrt @ L @ D_tilde_sqrt

            
            A_hat_khop = torch.FloatTensor(A_hat_khop).type(FloatTensor)
            A_hat_batch.append(A_hat_khop)

        A_hat = torch.block_diag(A_hat_batch[0])
        for k in range(1, batch_size):
            A_hat = torch.block_diag(A_hat, A_hat_batch[k])

        fixed_positions = torch.FloatTensor(fixed_positions).type(FloatTensor)
        fixed_positions_batch = torch.tile(fixed_positions, (batch_size,1))
        remain_positions = torch.FloatTensor(remain_positions).type(FloatTensor)
        remain_positions_batch = torch.tile(remain_positions, (batch_size,1))

        best_final_positions = 0
        best_loss = 1000000000000
        loss_ = 0

        best_final_k = 0
        best_final_epoch = 0

        # print("---------------------------------------")
        # print("start training GCN ... ")
        # print("=======================================")
        for train_step in range(1000):
            # print(train_step)
            # if train_step == 100:
            #     optimizer = SGD(gcn_network.parameters(), lr=0.0001, momentum=0.8, dampening=0, weight_decay=0.001, nesterov=False)

            final_positions_list = gcn_network(fixed_positions_batch, A_hat, num_remain)
            
            if dimension == 3:
                final_positions_list = [0.5 * torch.Tensor(np.array([config_width, config_length, config_height])).type(FloatTensor) * p for p in final_positions_list]
            else:
                final_positions_list = [0.5 * torch.Tensor(np.array([config_width, config_length])).type(FloatTensor) * p for p in final_positions_list]

            loss_list = []
            num_list = []

            loss = []

            # loss_step = []
            # loss_norm =[]

            for final_positions in final_positions_list:
                e_vals, num = check_number_of_clusters_torch(final_positions, config_communication_range)

                # final_positions_ = final_positions.cpu().data.numpy()
                # A = Utils.make_A_matrix(final_positions_, len(final_positions_), config_communication_range)
                # D = Utils.make_D_matrix(A, len(A))
                # L = D - A
                # flag, num_ = Utils.check_number_of_clusters(L, len(L))
                # print(num.cpu().data.numpy(), num_)
                num_list.append(num)

                loss_k = 5000*(num-1) + torch.max(torch.norm(final_positions-remain_positions, dim=1))# + torch.sum(torch.norm(final_positions-remain_positions, dim=1))/num_remain
                # loss_k = 5000*(num-1) + torch.sum(torch.norm(final_positions-remain_positions, dim=1))/num_remain
                loss_list.append(loss_k)
                # loss_step.append(torch.max(torch.norm(final_positions-remain_positions, dim=1)))
                loss.append(torch.sum(torch.norm(final_positions-remain_positions, dim=1))/num_remain)

            best_loss_k = min(loss_list)
            best_k = loss_list.index(best_loss_k)
            final_positions = final_positions_list[best_k]
            num = num_list[best_k]
            # print(num.cpu().data.numpy())

            final_positions_batch = torch.stack(final_positions_list).reshape(num_remain * batch_size, dimension)

            # loss
            # loss_step = torch.stack(loss_step)
            # weights_ones = torch.ones_like(loss_step)
            # loss_norm = torch.stack(loss_norm)

            # loss.append(loss_step @ weights_ones)
            # loss.append(loss_norm @ weights_ones)

            # loss1
            # loss.append(torch.max(torch.norm(final_positions-remain_positions, dim=1)))
            # loss.append(torch.sum(torch.norm(final_positions_batch-remain_positions_batch, dim=1))*10/num_remain)
            # loss.append(torch.norm(torch.norm(final_positions-remain_positions, dim=1)))
            # print(final_positions)
            # loss.append(torch.norm(e_vals))
            
            # print(loss)
            # loss_list += loss

            loss = torch.stack(loss_list)
            # print(loss)

            # initialization
            if train_step == 0:
                loss_weights = torch.ones_like(loss)
                # loss_weights = torch.tensor((1,1,10)).cuda()
                loss_weights = torch.nn.Parameter(loss_weights)
                T = loss_weights.sum().detach()
                loss_optimizer = Adam([loss_weights], lr=0.01)
                l0 = loss.detach()
                layer = gcn_network.gc8

            # compute the weighted loss
            # weighted_loss = 5000*(torch.sum(torch.tensor(num_list)) - batch_size) + loss_weights @ loss
            # weighted_loss = torch.exp(loss_weights-1) @ loss
            weighted_loss = loss_weights @ loss
            # weighted_loss = 1000 * (num - 1) + loss_weights @ loss + torch.var(degree-degree_init)
            # print(weighted_loss.cpu().data.numpy())
            if best_loss_k.cpu().data.numpy() < best_loss:
                best_loss = deepcopy(best_loss_k.cpu().data.numpy())
                best_final_positions = deepcopy(final_positions.cpu().data.numpy())
                best_final_k = best_k
                best_final_epoch = train_step

            # clear gradients of network
            optimizer.zero_grad()
            # backward pass for weigthted task loss
            weighted_loss.backward(retain_graph=True)
            # compute the L2 norm of the gradients for each task
            gw = []
            for i in range(len(loss)):
                dl = torch.autograd.grad(loss_weights[i]*loss[i], layer.parameters(), retain_graph=True, create_graph=True)[0]
                gw.append(torch.norm(dl))
            gw = torch.stack(gw)

            # compute loss ratio per task
            loss_ratio = loss.detach() / l0
            # compute the relative inverse training rate per task
            rt = loss_ratio / loss_ratio.mean()
            # compute the average gradient norm
            gw_avg = gw.mean().detach()
            # compute the GradNorm loss
            constant = (gw_avg * rt ** alpha).detach()
            gradnorm_loss = torch.abs(gw - constant).sum()
            # clear gradients of loss_weights
            loss_optimizer.zero_grad()
            # backward pass for GradNorm
            gradnorm_loss.backward()
            # log loss_weights and loss

            # update model loss_weights
            optimizer.step()
            # update loss loss_weights
            loss_optimizer.step()

            # renormalize loss_weights
            loss_weights = (loss_weights / loss_weights.sum() * T).detach()
            loss_weights = torch.nn.Parameter(loss_weights)
            loss_optimizer = torch.optim.Adam([loss_weights], lr=0.01)
            
            loss_ = best_loss_k.cpu().data.numpy()
            print(f"episode {train_step}, num {num.cpu().data.numpy()}, loss {loss_:.6f}", end='\r')
            # print(f"episode {train_step}, num {num.cpu().data.numpy()}, loss {loss_}, weights {loss_weights.cpu().data.numpy()}", end='\r')
            # print(torch.norm(final_positions[max_index] - remain_positions[max_index]), torch.norm(centroid - centrepoint))

            if draw and train_step % 200 == 0:
                remain_positions_ = remain_positions.cpu().data.numpy()
                final_positions_ = final_positions.cpu().data.numpy()

                plt.scatter(remain_positions_[:,0], remain_positions_[:,1], c='black')
                plt.scatter(final_positions_[:,0], final_positions_[:,1], c='g')
                plt.text(10, 10, f'best loss: {loss_}')
                # plt.xlim(0, 1000)
                # plt.ylim(0, 1000)
                plt.show()
            

        speed = np.zeros((config_num_of_agents, dimension))
        remain_positions_numpy = remain_positions.cpu().data.numpy()
        temp_max_distance = 0
        # print("=======================================")

        for i in range(num_remain):
            if np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]) > 0:
                speed[remain_list[i]] = (best_final_positions[i] - remain_positions_numpy[i]) / np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i])
            if np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]) > temp_max_distance:
                temp_max_distance = deepcopy(np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]))

        max_time = temp_max_distance / config_constant_speed

        print(f"trained: max time {max_time}, best episode {best_final_epoch}, best k-hop {best_final_k+2}")

        if draw:
            remain_positions_ = remain_positions.cpu().data.numpy()

            plt.scatter(remain_positions_[:,0], remain_positions_[:,1], c='black')
            plt.scatter(best_final_positions[:,0], best_final_positions[:,1], c='g')
            plt.text(10, 10, f'best time: {max_time}')
            plt.xlim(0, 1000)
            plt.ylim(0, 1000)
            plt.show()
        # print(max_time)

        return deepcopy(speed), deepcopy(max_time), deepcopy(best_final_positions)


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def make_dd_khop_A_matrix(all_neighbor, positions, num_remain, khop=5):
    # all_neighbor = calculate_khop_neighbour(positions, d)

    num_of_agents = len(positions)
    A = np.zeros((num_of_agents, num_of_agents))

    for i in range(num_remain):
        for j in range(num_remain, num_of_agents-1):
            hop = 1
            # if np.linalg.norm(positions[j] - central_point) >= embedding_distence : continue
            # dis = np.linalg.norm(positions[j] - central_point) / embedding_distence 
            # dis = np.cos(dis*np.pi/2)
            while(j not in all_neighbor[i][hop-1]):hop+=1
            if hop <= khop:
                A[i, j] = 1
                A[j, i] = 1

    for i in range(num_of_agents-1):
        A[i, num_of_agents-1] = 1
        A[num_of_agents-1, i] = 1

    # print(A)
    return deepcopy(A)

def calculate_khop_neighbour(positions, d):
    num_of_agents = len(positions)
    neighbour_i_1hop = []
    for i in range(num_of_agents):
        neighbour = []
        for j in range(num_of_agents):
            if i==j: continue

            if np.linalg.norm(positions[i, :] - positions[j, :]) <= d:
                neighbour.append(j)

        neighbour_i_1hop.append(deepcopy(neighbour))

    all_neighbour = []
    for i in range(num_of_agents):
        cnt = 0
        neighbour_i_all = []
        neighbour_i_multihop = deepcopy(neighbour_i_1hop[i])

        while len(neighbour_i_multihop) > 0:
            neighbour_i_all.append(deepcopy(neighbour_i_multihop))
            neighbour_i_multihop = []

            for j in neighbour_i_all[cnt]:
                for k in neighbour_i_1hop[j]:
                    if k not in sum(neighbour_i_all, []) and k not in neighbour_i_multihop and k != i:
                        neighbour_i_multihop.append(k)

            cnt += 1
            
        all_neighbour.append(deepcopy(neighbour_i_all))

    return deepcopy(all_neighbour)


class GCN_dd_structure(nn.Module):
    def __init__(self, nfeat=3, nhid=5, nclass=3, dropout=0.5, if_dropout=True, bias=True):
        super(GCN_dd_structure, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, bias=bias)
        self.gc2 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc3 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc4 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc5 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc6 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc7 = GraphConvolution(nhid, nhid, bias=bias)
        self.gc8 = GraphConvolution(nhid, nclass, bias=bias)
        self.dropout = dropout
        self.training = if_dropout

    def forward(self, x, adj, num_remain):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = self.gc8(x, adj)
        x = x.split(config_num_of_agents+1, dim=0)
        x = [torch.tanh(x_remain.split(num_remain, dim=0)[0])+1 for x_remain in x]
        # x = [x_remain.split(num_remain, dim=0)[0] for x_remain in x]

        # return torch.tanh(x) + 1
        return x
    
def check_number_of_clusters_torch(positions, d):
    m, n = positions.shape
    G = torch.mm(positions, positions.T)
    H = torch.tile(torch.diag(G), (m, 1))
    D = torch.sqrt(H + H.T - 2*G)
    # print(m, n, D)
    A = torch.where(D > d, 0, 1.0)
    # print(m, n, A)
    D = torch.diag(torch.sum(A, dim=1))
    # print(m, n, D)
    L = D - A

    try:
        e_vals, e_vecs = torch.linalg.eigh(L)
    except:
        e_vals, e_vecs = np.linalg.eig(L.cpu().data.numpy())
        e_vals = torch.FloatTensor(e_vals.real).type(torch.cuda.FloatTensor)
    # print(e_vals.real)
        
    num = torch.sum(torch.where(e_vals.real < 0.0001, 1, 0))
    # print(torch.sum(e_vals), torch.trace(L), torch.sum(A))
    return e_vals.real, num