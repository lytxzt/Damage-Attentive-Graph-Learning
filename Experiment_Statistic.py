
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from copy import deepcopy
import time

from Environment import Environment
from Swarm import Swarm
from Configurations import *
import Utils

max_step = int(0.5 * config_width)
# max_step = 1

use_pretrained = False
draw = False

numpy_speedup = True

# for dnum in range(10, 180, 10):
for dnum in [10]:
    for mode in [3]:
    # for mode in [3]:
        config_algorithm_mode = mode
        algorithm_mode = {1: "HERO",
                          2: "CEN",
                          3: "SIDR",
                          4: "GCN_2017",
                          5: "GAT",
                          6: "CR-MGC",
                          7: "DEMD",
                          8: "DAGL"}

        print("CNS issue Starts...")
        print("------------------------------")
        print(f"Algorithm: {algorithm_mode[config_algorithm_mode]} with {dnum} UAVs destructed")

        # storage        
        storage_random_seed = []
        storage_connect_step = []
        storage_connect_positons = []
        storage_node_degree = []

        # change the number of destructed UAVs
        config_num_destructed_UAVs = dnum  # should be in the range of [1, config_num_-2]

        # change the seed to alternate the UED
        # seed = [61, 29, 83, 3, 59, 22, 8, 96, 80, 20, 39, 19, 89, 75, 79, 55, 61, 74, 8, 89, 83, 3, 38, 88, 56, 68, 67, 46, 48, 63, 54, 43, 52, 72, 75, 21, 64, 44, 50, 77, 39, 14, 18, 66, 82, 51, 65, 90, 57, 35, 92, 74, 9, 64, 52, 91, 56, 87, 77, 82, 34, 38, 38, 97, 3, 85, 67, 15, 86, 21, 67, 68, 98, 99, 32, 100, 48, 19, 85, 67, 81, 25, 64, 23, 37, 87, 31, 8, 96, 47, 63, 57, 1, 66, 18, 89, 54, 60, 58, 25]
        seed = config_random_seed

        case = 0
        max_case = 50
        # for case in range(60):
        while len(storage_random_seed) < max_case and case < len(seed):
            environment = Environment()
            
            swarm = Swarm(algorithm_mode=config_algorithm_mode, use_pretrained=use_pretrained)
            num_cluster_list = []

            environment_positions = environment.reset()
            swarm.reset()

            np.random.seed(seed[case]+1718)
            random.seed(seed[case]+1718)
            # print(seed[case])

            # flag if break the CCN
            break_CCN_flag = True
            # num of connected steps
            num_connected_steps = 0

            destroy_num, destroy_list = environment.stochastic_destroy(mode=2, num_of_destroyed=config_num_destructed_UAVs)
            swarm.destroy_happens(deepcopy(destroy_list), deepcopy(environment_positions))

            initial_remain_positions = []
            destroy_positions = []
            for i in range(config_num_of_agents):
                if i in destroy_list:
                    destroy_positions.append(deepcopy(config_initial_swarm_positions[i]))
                else:
                    initial_remain_positions.append(deepcopy(config_initial_swarm_positions[i]))
            initial_remain_positions = np.array(initial_remain_positions)
            destroy_positions = np.array(destroy_positions)
            A = Utils.make_A_matrix(initial_remain_positions, config_num_of_agents - config_num_destructed_UAVs, config_communication_range)
            num_cluster = environment.check_the_clusters()
            
            # check if the UED break the CCN of the USNET
            if num_cluster == 1:
                # print(f"case {case}: not distructed")
                case += 1
                continue

            print("=======================================")

            positions_with_clusters = Utils.split_the_positions_into_clusters(initial_remain_positions, num_cluster, A)

            if numpy_speedup:
                step, final_positions, num = swarm.solve()
                print(f"case {case}: step {step} ---num of sub-nets {num}")
            else:
                for step in range(max_step):
                    actions, max_time, _ = swarm.take_actions()
                    # actions, max_time, time_consum = swarm.take_actions()
                    # with open(f"./Logs/time/{algorithm_mode[config_algorithm_mode]}.txt", 'a') as f:
                    #     print(time_consum, file=f)

                    environment_next_positions = environment.next_state(deepcopy(actions))
                    swarm.update_true_positions(environment_next_positions)

                    temp_cluster = environment.check_the_clusters()
                    num_cluster_list.append(temp_cluster)
                    # print("---------------------------------------")
                    if temp_cluster == 1:
                        print(f"case {case}: step {step} ---num of sub-nets {environment.check_the_clusters()}")
                    else:
                        num_connected_steps += 1
                        print(f"case {case}: step {step} ---num of sub-nets {environment.check_the_clusters()}", end='\r')
                        if step == max_step-1:
                            print(f"case {case}: step {step} ---num of sub-nets {environment.check_the_clusters()}")

                    if environment.check_the_clusters() == 1:
                        break

                    # update
                    environment.update()
                    environment_positions = deepcopy(environment_next_positions)

                # print("case %d: step %d ---num of clusters %d -- connected" % (case, step, environment.check_the_clusters()))
                
                final_positions = np.take(environment_positions, swarm.remain_list, axis=0)


            final_positions = np.array(final_positions)

            storage_random_seed.append(case)
            storage_connect_step.append(step)
            storage_connect_positons.append(final_positions)

            A = Utils.make_A_matrix(final_positions, len(final_positions), config_communication_range)
            degrees = [int(d) for d in np.sum(A, axis=0)]

            storage_node_degree.extend(deepcopy(degrees))
            
            print(f"avg: {np.mean(np.array(storage_connect_step))}")

            case += 1

            if draw:
                plt.scatter(initial_remain_positions[:,0], initial_remain_positions[:,1], c='black')
                plt.scatter(final_positions[:,0], final_positions[:,1], c='g')
                plt.text(10, 10, f'best time: {max_time}')
                plt.xlim(0, 1000)
                plt.ylim(0, 1000)
                plt.show()

        with open(f'./Logs/scalable/N{config_num_of_agents}/{algorithm_mode[config_algorithm_mode]}_d{config_num_destructed_UAVs}.txt', 'w') as f:
        # with open(f'./Logs/damage/d{dnum}/{algorithm_mode[config_algorithm_mode]}_d{dnum}.txt', 'w') as f:
            print('case:\n', storage_random_seed, file=f)
            print('\nconnect_step:\n', storage_connect_step, file=f)
            print('\navg_connect_step:\n', np.mean(np.array(storage_connect_step)), file=f)
            print('\ndegree_distribution:\n', storage_node_degree, file=f)
            for i in range(len(storage_random_seed)):
                print('\n=======================================', file=f)
                print(f'case {storage_random_seed[i]} -- step {storage_connect_step[i]} -- connected', file=f)
                print('final positions:\n', storage_connect_positons[i], file=f)
