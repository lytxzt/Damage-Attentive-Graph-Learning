import Utils
from copy import deepcopy
from Configurations import *
from Environment import Environment
import matplotlib.pyplot as plt

khop = 5

class SwarmLocal:
    def __init__(self):
        self.initial_positions = deepcopy(config_initial_swarm_positions)
        self.remain_list = [i for i in range(config_num_of_agents)]
        self.remain_num = config_num_of_agents
        self.num_of_agents = config_num_of_agents
        self.max_destroy_num = config_maximum_destroy_num

        self.remain_positions = deepcopy(self.initial_positions)
        self.true_positions = deepcopy(self.initial_positions)
        
        self.khop_neighbour = []
        self.calculate_khop_neighbour()

    def destroy_happens(self, destroy_list, environment_positions):
        self.notice_destroy = True
        for destroy_index in destroy_list:
            self.remain_list.remove(destroy_index)
        self.true_positions = deepcopy(environment_positions)
        self.remain_num = len(self.remain_list)

        self.make_remain_positions()
        self.calculate_khop_neighbour()

    def make_remain_positions(self):
        self.remain_positions = []
        for i in self.remain_list:
            self.remain_positions.append(deepcopy(self.true_positions[i]))
        self.remain_positions = np.array(self.remain_positions)

    def update_true_positions(self, environment_positions):
        self.true_positions = deepcopy(environment_positions)

    def calculate_khop_neighbour(self, khop=khop):
        neighbour_i_1hop = []
        for i in range(self.num_of_agents):
            neighbour = []

            if i in self.remain_list:
                for j in range(self.num_of_agents):
                    if i==j: continue
                    if euclidean(self.true_positions[i], self.true_positions[j]) <= config_communication_range:
                        neighbour.append(j)

            neighbour_i_1hop.append(deepcopy(neighbour))

        self.khop_neighbour = []
        for i in range(self.num_of_agents):
            neighbour_i_all = []

            if i in self.remain_list:
                neighbour_i_multihop = deepcopy(neighbour_i_1hop[i])
                
                for cnt in range(khop):
                    neighbour_i_all.append(deepcopy(neighbour_i_multihop))
                    neighbour_i_multihop = []

                    for j in neighbour_i_all[cnt]:
                        for k in neighbour_i_1hop[j]:
                            if k not in sum(neighbour_i_all, []) and k not in neighbour_i_multihop and k != i:
                                neighbour_i_multihop.append(k)
                
            # print(neighbour_i_all)
            self.khop_neighbour.append(deepcopy(neighbour_i_all))

    def detect_actions(self, alg=0):
        detect_flag_list = []
        e_lamda_list = []

        for node in self.remain_list:
            khop_positions = [self.true_positions[node]]
            for i in sum(self.khop_neighbour[node], []):
                khop_positions.append(self.true_positions[i])

            # detect split with laplace matrix alg
            if alg == 0:
                khop_positions = np.array(khop_positions)

                A = Utils.make_A_matrix(khop_positions, len(khop_positions), config_communication_range)
                D = Utils.make_D_matrix(A, len(khop_positions))
                L = D - A
                e_vals, e_vecs = np.linalg.eig(L)
                e_lamda = np.sort(e_vals)[1]
                e_lamda = e_lamda if e_lamda >= 0.000001 else 0
                print(node, e_lamda)

                e_lamda_list.append(e_lamda)
                detect_flag_list.append(e_lamda==0)
                # connected_flag, num_of_clusters = Utils.check_number_of_clusters(L, len(khop_positions))
                # print(num_of_clusters)

        fig = plt.figure()
        plt.plot([i for i in range(len(e_lamda_list))], e_lamda_list)
        plt.show()

        print(len([i for i in detect_flag_list if not i]))





def euclidean(x, y):
    return np.sqrt(np.sum((x - y)**2))

    
if __name__ == "__main__":
    np.random.seed(57)
    random.seed(77)

    environment = Environment()
    swarm = SwarmLocal()

    environment_positions = environment.reset()
    destroy_num, destroy_list = environment.stochastic_destroy(mode=2, num_of_destroyed=100)

    swarm.destroy_happens(deepcopy(destroy_list), deepcopy(environment_positions))

    swarm.detect_actions()






