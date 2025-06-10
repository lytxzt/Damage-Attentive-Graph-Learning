from copy import deepcopy
import time

import Utils
from Configurations import *
from Environment import Environment

from Previous_Algorithm.CR_MGC import CR_MGC
from Previous_Algorithm.DEMD import DEMD
from Previous_Algorithm.GCN_2017 import GCN_2017
from Previous_Algorithm.GAT import GAT
from Previous_Algorithm.Centering import *
from Previous_Algorithm.SIDR import SIDR
from Previous_Algorithm.HERO import HERO

from DAGL_Algorithm.DAGL_Framework import DAGL


class Swarm:
    def __init__(self, algorithm_mode=1, use_pretrained=False, khop=3):
        self.initial_positions = deepcopy(config_initial_swarm_positions)
        self.remain_list = [i for i in range(config_num_of_agents)]
        self.remain_num = config_num_of_agents
        self.num_of_agents = config_num_of_agents
        self.max_destroy_num = config_maximum_destroy_num

        self.remain_positions = deepcopy(self.initial_positions)
        self.true_positions = deepcopy(self.initial_positions)
        
        self.algorithm_mode = algorithm_mode

        self.if_once_gcn = False
        self.once_destroy_gcn_speed = np.zeros((self.num_of_agents, config_dimension))
        self.max_time = 0

        self.khop = khop
        self.demd = DEMD()

        self.ddag = DAGL(use_pretrained=use_pretrained)
        
        self.cr_gcm = CR_MGC(use_meta=False)
        self.gcn_2017 = GCN_2017()
        self.gat = GAT()

        self.hero = HERO(self.initial_positions)

        self.if_once_gcn_network = False
        self.once_destroy_gcn_network_speed = np.zeros((self.num_of_agents, config_dimension))

        self.best_final_positions = 0

        self.notice_destroy = False
        self.destination_positions = np.zeros((self.num_of_agents, config_dimension))
        self.inertia_counter = 0
        self.inertia = 100
        self.if_finish = [True for i in range(self.num_of_agents)]

    def destroy_happens(self, destroy_list, environment_positions):
        self.notice_destroy = True
        for destroy_index in destroy_list:
            self.remain_list.remove(destroy_index)
        self.true_positions = deepcopy(environment_positions)
        self.remain_num = len(self.remain_list)
        # self.csds.notice_destroy(deepcopy(destroy_list))

    def update_true_positions(self, environment_positions):
        self.true_positions = deepcopy(environment_positions)

    def reset(self, change_algorithm_mode=False, algorithm_mode=0):
        self.remain_list = [i for i in range(config_num_of_agents)]
        self.remain_num = config_num_of_agents
        self.positions = []
        self.mean_positions = []
        self.target_positions = []
        self.max_time = 0

        if change_algorithm_mode:
            self.algorithm_mode = algorithm_mode

        self.if_once_gcn = False
        self.once_destroy_gcn_speed = np.zeros((self.num_of_agents, config_dimension))

        self.if_once_gcn_network = False
        self.once_destroy_gcn_network_speed = np.zeros((self.num_of_agents, config_dimension))

    def take_actions(self):
        """
        take actions with global information (GI)
        :return: unit speed vectors
        """
        actions = np.zeros((self.num_of_agents, config_dimension))
        max_time = 0
        self.make_remain_positions()
        flag, num_cluster = Utils.check_if_a_connected_graph(deepcopy(self.remain_positions), len(self.remain_list))

        time_start = time.time()
        if flag:
            # print("connected")
            return deepcopy(actions), max_time
        else:
            # HERO
            if self.algorithm_mode == 1:
                actions_hero = self.hero.hero(
                    Utils.difference_set([i for i in range(self.num_of_agents)], self.remain_list), self.true_positions)

                # for i in self.remain_list:
                #     actions[i] = 0.2 * centering_fly(self.true_positions, self.remain_list, i) + 0.8 * actions_hero[i]
                actions = 0.2 * centering_fly_v2(self.true_positions, self.remain_list) + 0.8 * actions_hero


            # centering
            elif self.algorithm_mode == 2:
                # for i in self.remain_list:
                    # actions[i] = centering_fly(self.true_positions, self.remain_list, i)
                actions = centering_fly_v2(self.true_positions, self.remain_list)


            # SIDR
            elif self.algorithm_mode == 3:
                actions = SIDR(self.true_positions, self.remain_list)


            # GCN
            elif self.algorithm_mode == 4:
                if self.if_once_gcn_network:
                    for i in range(len(self.remain_list)):
                        if np.linalg.norm(
                                self.true_positions[self.remain_list[i]] - self.best_final_positions[i]) >= 0.55:
                            actions[self.remain_list[i]] = deepcopy(
                                self.once_destroy_gcn_network_speed[self.remain_list[i]])
                        # else:
                        #     print("%d already finish" % self.remain_list[i])
                    max_time = deepcopy(self.max_time)
                else:
                    self.if_once_gcn_network = True
                    actions, max_time, best_final_positions = self.gcn_2017.gcn(deepcopy(self.true_positions),
                                                                                     deepcopy(self.remain_list))
                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    self.best_final_positions = deepcopy(best_final_positions)
                    self.max_time = deepcopy(max_time)


            # GAT
            elif self.algorithm_mode == 5:
                if self.if_once_gcn_network:
                    for i in range(len(self.remain_list)):
                        if np.linalg.norm(
                                self.true_positions[self.remain_list[i]] - self.best_final_positions[i]) >= 0.55:
                            actions[self.remain_list[i]] = deepcopy(
                                self.once_destroy_gcn_network_speed[self.remain_list[i]])
                        # else:
                        #     print("%d already finish" % self.remain_list[i])
                    max_time = deepcopy(self.max_time)
                else:
                    self.if_once_gcn_network = True
                    actions, max_time, best_final_positions = self.gat.gat(deepcopy(self.true_positions),
                                                                                     deepcopy(self.remain_list))
                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    self.best_final_positions = deepcopy(best_final_positions)
                    self.max_time = deepcopy(max_time)

            
            # CR-MGC
            elif self.algorithm_mode == 6:
                if self.if_once_gcn_network:
                    for i in range(len(self.remain_list)):
                        if np.linalg.norm(self.true_positions[self.remain_list[i]] - self.best_final_positions[i]) >= 0.55:
                            actions[self.remain_list[i]] = deepcopy(self.once_destroy_gcn_network_speed[self.remain_list[i]])

                    max_time = deepcopy(self.max_time)
                else:
                    self.if_once_gcn_network = True
                    actions, max_time, best_final_positions = self.cr_gcm.cr_gcm(deepcopy(self.true_positions),
                                                                                 deepcopy(self.remain_list))
                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    self.best_final_positions = deepcopy(best_final_positions)
                    self.max_time = deepcopy(max_time)


            # DEMD
            elif self.algorithm_mode == 7:
                if self.if_once_gcn_network:
                    for i in range(len(self.remain_list)):
                        d = np.linalg.norm(self.true_positions[self.remain_list[i]] - self.best_final_positions[i])
                        if d >= 1:
                            actions[self.remain_list[i]] = deepcopy(self.once_destroy_gcn_network_speed[self.remain_list[i]])
                        elif d > 0.0001:
                            actions[self.remain_list[i]] = deepcopy(self.best_final_positions[i] - self.true_positions[self.remain_list[i]])
                    max_time = deepcopy(self.max_time)
                else:
                    self.if_once_gcn_network = True
                    # actions, max_time, best_final_positions = self.demd.demd(deepcopy(self.true_positions), deepcopy(self.remain_list), self.khop)
                    actions, max_time, best_final_positions = self.demd.demd_adaptive(deepcopy(self.true_positions), deepcopy(self.remain_list))
                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    self.best_final_positions = deepcopy(best_final_positions)
                    self.max_time = deepcopy(max_time)
                    
            
            # proposed ML-DAGL algorithm
            elif self.algorithm_mode == 8:
                if self.if_once_gcn_network:
                    for i in range(len(self.remain_list)):
                        d = np.linalg.norm(self.true_positions[self.remain_list[i]] - self.best_final_positions[i])
                        if d >= 1:
                            actions[self.remain_list[i]] = deepcopy(self.once_destroy_gcn_network_speed[self.remain_list[i]])
                        elif d > 0.0001:
                            actions[self.remain_list[i]] = deepcopy(self.best_final_positions[i] - self.true_positions[self.remain_list[i]])

                    max_time = deepcopy(self.max_time)
                else:
                    self.if_once_gcn_network = True
                    actions, max_time, best_final_positions = self.ddag.dagl(deepcopy(self.true_positions), deepcopy(self.remain_list))
                    self.once_destroy_gcn_network_speed = deepcopy(actions)
                    self.best_final_positions = deepcopy(best_final_positions)
                    self.max_time = deepcopy(max_time)

            else:
                print("No such algorithm")

        time_end = time.time()
        # with open(f"./Logs/time/{self.algorithm_mode}.txt", 'a') as f:
        #     print(time_end - time_start, file=f)

        return deepcopy(actions), deepcopy(self.max_time), deepcopy(self.best_final_positions)

    def make_remain_positions(self):
        self.remain_positions = []
        for i in self.remain_list:
            self.remain_positions.append(deepcopy(self.true_positions[i]))
        self.remain_positions = np.array(self.remain_positions)

    def check_if_finish(self, cluster_index):
        flag = True
        for i in range(len(cluster_index)):
            if not self.if_finish[self.remain_list[cluster_index[i]]]:
                flag = False
                break
        return flag
    
    def check_number_of_clusters(self):
        m, n = self.remain_positions.shape
        G = np.matmul(self.remain_positions, self.remain_positions.T)
        H = np.tile(np.diag(G), (m, 1))
        D = np.sqrt(H + H.T - 2*G)
        # print(m, n, D)
        A = np.where(D > config_communication_range, 0, 1.0)
        # print(m, n, A)
        D = np.diag(np.sum(A, axis=1))
        # print(m, n, D)
        L = D - A

        e_vals, e_vecs = np.linalg.eigh(L)
        # print(e_vals.real)
            
        num = np.sum(np.where(e_vals.real < 0.000001, 1, 0))
        # print(torch.sum(e_vals), torch.trace(L), torch.sum(A))
        return e_vals.real, num

    def solve(self):
        # self.make_remain_positions()
        # speed = torch.zeros_like(actions)
        max_step = int(0.5 * config_width)
        num_of_remain_agents = len(self.remain_list)

        positions = deepcopy(self.remain_positions)

        for step in range(max_step):
            actions, _, _ = self.take_actions()
            # actions = np.take(actions, self.remain_list, axis=0)
            positions = self.true_positions + actions * config_constant_speed

            self.update_true_positions(positions)
            self.make_remain_positions()

            # A = Utils.make_A_matrix(self.remain_positions, num_of_remain_agents, config_communication_range)
            # D = Utils.make_D_matrix(A, num_of_remain_agents)
            # L = D - A
            # connected_flag, num = self..check_number_of_clusters(L, num_of_remain_agents)
            connected_flag, num = self.check_number_of_clusters()

            if num == 1:
                break

            print(f"step {step} ---num of sub-nets {num}", end='\r')

        return step, self.remain_positions, num



if __name__ == "__main__":
    np.random.seed(57)
    random.seed(77)

    environment = Environment()
    swarm = Swarm()

    environment_positions = environment.reset()
    destroy_num, destroy_list = environment.stochastic_destroy(mode=2, num_of_destroyed=100)

    swarm.destroy_happens(deepcopy(destroy_list), deepcopy(environment_positions))

    print(len(swarm.remain_list))
