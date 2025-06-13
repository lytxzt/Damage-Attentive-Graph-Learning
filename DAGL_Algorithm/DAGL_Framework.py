import torch
from torch.optim import Adam
from copy import deepcopy

import Utils
from DAGL_Algorithm.Bipartite_GCN import Bipartite_GCN_structure
from DAGL_Algorithm.Damage_Attention import dilated_DAM
from Configurations import *

best_hidden_dimension = 512
best_dropout = 0.1

save_loss_curve = False
use_input_norm = True

torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

class DAGL:
    def __init__(self, use_pretrained=True):
        self.hidden_dimension = best_hidden_dimension
        self.dropout_value = best_dropout
        self.use_pretrained = use_pretrained

    def dagl(self, global_positions, remain_list, khop=1, batch_size=10):
        gcn_network = Bipartite_GCN_structure(nfeat=config_dimension, nhid=self.hidden_dimension, nclass=config_dimension, dropout=self.dropout_value, if_dropout=True, bias=True)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            gcn_network.cuda()

        if self.use_pretrained:
            gcn_network.load_state_dict(torch.load(f"./Pretrained_model/model_N{config_num_of_agents}.pt"))

        # self.optimizer = Adam(self.gcn_network.parameters(), lr=0.001)
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        optimizer = Adam(gcn_network.parameters(), lr=0.0001)
        # optimizer = SGD(gcn_network.parameters(), lr=0.01, momentum=0.9)

        num_of_agents = len(global_positions)
        num_remain = len(remain_list)
        num_destructed = num_of_agents - num_remain

        # prepare data
        remain_positions, fixed_positions, A_hat = dilated_DAM(global_positions, remain_list, khop, batch_size)

        # input normalization
        if use_input_norm:
            fixed_positions = (fixed_positions - config_central_point) / 500
        
        # cuda
        remain_positions = torch.FloatTensor(remain_positions).type(FloatTensor)
        fixed_positions = torch.FloatTensor(fixed_positions).type(FloatTensor)
        A_hat = torch.FloatTensor(A_hat).type(FloatTensor)

        # initialize training
        best_final_positions = 0
        best_loss = 1000000000000
        loss_ = 0

        best_final_k = 0
        best_final_epoch = 0

        loss_storage = []
        num_storage = []

        # algorithm start
        for train_step in range(50):
            # early stop
            # if loss_ > 10000 and train_step > 50:
            #     break

            final_positions_list = gcn_network(fixed_positions, A_hat, num_remain)
            
            # output extending
            if use_input_norm:
                final_positions_list = [0.5 * torch.Tensor(config_space_range).type(FloatTensor) * (torch.tanh(p) + 1) for p in final_positions_list]

            loss_list = []
            num_list = []

            for final_positions in final_positions_list:
                try:
                    e_vals, num = Utils.check_number_of_clusters_torch(final_positions, config_communication_range)
                except:
                    final_positions_ = final_positions.cpu().data.numpy()
                    A = Utils.make_A_matrix(final_positions_, len(final_positions_), config_communication_range)
                    D = Utils.make_D_matrix(A, len(A))
                    L = D - A
                    flag, num = Utils.check_number_of_clusters(L, len(L))
                    
                num_list.append(num)

                # calculate loss
                loss_k = 5000*(num-1) + torch.max(torch.norm(final_positions-remain_positions, dim=1))
                loss_list.append(loss_k)

            # calculate dilated loss
            loss = torch.stack(loss_list)
            
            loss_weights = torch.nn.Parameter(torch.ones_like(loss))
            loss = loss_weights @ loss

            # iteration
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            optimizer.step()

            # find best solution
            best_loss_k = min(loss_list)
            best_k = loss_list.index(best_loss_k)
            final_positions = final_positions_list[best_k]
            num = num_list[best_k]
            
            loss_ = best_loss_k.cpu().data.numpy()
            num_ = num.cpu().data.numpy() if torch.is_tensor(num) else num
            print(f"episode {train_step}, num {num_}, loss {loss_:.6f}", end='\r')

            if loss_ < best_loss:
                best_loss = deepcopy(loss_)
                best_final_positions = deepcopy(final_positions.cpu().data.numpy())
                best_final_k = best_k
                best_final_epoch = train_step
            
            # storage
            num_storage.append(int(num_))
            loss_storage.append(loss_ % 5000)
            
            # if train_step > 500 and num_ == 1:
            #     torch.save(gcn_network.state_dict(), f"./Pretrained_model/model_N{config_num_of_agents}.pt")
         
        # update speed
        speed = np.zeros((config_num_of_agents, config_dimension))
        remain_positions_numpy = remain_positions.cpu().data.numpy()
        temp_max_distance = 0

        for i in range(num_remain):
            if np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]) > 0:
                speed[remain_list[i]] = (best_final_positions[i] - remain_positions_numpy[i]) / np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i])
            if np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]) > temp_max_distance:
                temp_max_distance = deepcopy(np.linalg.norm(best_final_positions[i] - remain_positions_numpy[i]))

        # speed = speed / np.max(np.linalg.norm(speed, axis=1))

        max_time = temp_max_distance / config_constant_speed

        print(f"trained: max time {max_time}, best episode {best_final_epoch}, best k-hop {best_final_k+1}")
            
        if save_loss_curve:
            with open(f'./Logs/loss/loss_d{num_destructed-1}_setup.txt', 'a') as f:
                print(loss_storage, file=f)

            with open(f'./Logs/loss/num_d{num_destructed-1}_setup.txt', 'a') as f:
                print(num_storage, file=f)

        return deepcopy(speed), deepcopy(max_time), deepcopy(best_final_positions)

    
