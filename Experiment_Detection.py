from Environment import Environment
from Swarm_delay import SwarmDelay
from Configurations import *
import Utils
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from Drawing.Draw_Static import *
from Main_algorithm_GCN.GCO import GCO



print("Split Detection problem Starts...")
print("------------------------------")

environment = Environment()
swarm = SwarmDelay()
num_cluster_list = []

environment_positions = environment.reset()
swarm.calculate_neighbour()
neighbour_list = deepcopy(swarm.all_neighbour)

# storage
storage_remain_list = []
storage_positions = []
# storage_destroy_positions = []
storage_detection_states = []
storage_remain_detection_matrix = []

break_neighbour_list = []
remain_neighbour_list = []

# change the number of destructed UAVs
config_num_destructed_UAVs = 150  # should be in the range of [1, config_num_-2]
max_step = 25

# change the seed to alternate the UED
# np.random.seed(17)
# random.seed(18)
np.random.seed(57)
random.seed(77)

# destruction
storage_remain_list.append(deepcopy(swarm.remain_list))
storage_positions.append(deepcopy(swarm.true_positions))
# storage_destroy_positions.append([])
storage_detection_states.append(False)
# storage_remain_detection_matrix.append(
#     deepcopy(Utils.make_A_matrix(swarm.true_positions, config_num_of_agents, config_communication_range)))

# num of connected steps
num_connected_steps = 0

# draw figs
break_positions = []
remain_positions_detect = []
remain_positions_undetect = []
fig = plt.figure()

for step in range(max_step):
    # flag if detect the break
    detect_break_flag = False
    detect_num = 0

    if step == 0:
        print("=======================================")
        print(f"destroy {0} -- mode {2} num {config_num_destructed_UAVs} ")
        destroy_num, destroy_list = environment.stochastic_destroy(mode=2, num_of_destroyed=config_num_destructed_UAVs)
        # destroy_list = [64, 83, 50, 61, 49, 29, 74, 121, 142, 156, 60, 37, 191, 161, 169, 128, 0, 71, 7, 127, 44, 48, 129, 99, 46, 84, 82, 30, 22, 141, 66, 136, 120, 52, 57, 195, 18, 20, 51, 143, 112, 152, 137, 19, 25, 54, 174, 171, 24, 147, 118, 162, 157, 27, 69, 113, 170, 31, 89, 135, 14, 103, 140, 63, 56, 167, 138, 35, 98, 3, 8, 13, 126, 80, 5, 95, 45, 97, 55, 15, 32, 199, 115, 72, 88, 62, 181, 198, 189, 105, 117, 159, 12, 163, 38, 67, 11, 2, 114, 123, 96, 53, 73, 183, 144, 196, 106, 130, 160, 1, 172, 133, 85, 59, 139, 164, 111, 184, 17, 91, 33, 26, 109, 148, 21, 190, 92, 6, 194, 149, 178, 131, 101, 70, 34, 173, 108, 39, 78, 93, 65, 81, 168, 182, 192, 107, 90, 188, 187, 4]
        print(f"destroy {config_num_destructed_UAVs} nodes \ndestroy index list :")
        print(destroy_list)
        swarm.destroy_happens(deepcopy(destroy_list), deepcopy(environment_positions))

        print("remain index list :")
        print(swarm.remain_list)

        break_positions = [[environment_positions[n][0] for n in destroy_list], [environment_positions[n][1] for n in destroy_list]]

        initial_remain_positions = []
        destroy_positions = []
        for i in range(config_num_of_agents):
            if i in destroy_list:
                destroy_positions.append(deepcopy(config_initial_swarm_positions[i]))
            else:
                initial_remain_positions.append(deepcopy(config_initial_swarm_positions[i]))

        initial_remain_positions = np.array(initial_remain_positions)
        destroy_positions = np.array(destroy_positions)
        A = Utils.make_A_matrix(initial_remain_positions, config_num_of_agents - config_num_destructed_UAVs,
                                config_communication_range)
        num_cluster = environment.check_the_clusters()
        print(f"cluster num : {num_cluster}")

        # check if the UED break the CCN of the USNET
        if num_cluster == 1:
            print("---------------------------------------")
            print("This kind of UED does not break the CCN of the USNET!")
            print("Please change the random seed or the number of destructed UAVs!")
            print("Algorithm Ends")
            print("---------------------------------------")
            break_CCN_flag = False
            break

        # store the 1hop break neighbour and remain neighbour list for each node
        break_1hop_neighbour = []
        remain_1hop_neighbour = []

        for node in range(config_num_of_agents):
            break_neighbour, remain_neighbour = [], []

            if node in swarm.remain_list:
                for hop_neighbour in neighbour_list[node][0]:
                    if hop_neighbour in destroy_list:
                        break_neighbour.append(hop_neighbour)
                    else:
                        remain_neighbour.append(hop_neighbour)

            break_1hop_neighbour.append(deepcopy(break_neighbour))
            remain_1hop_neighbour.append(deepcopy(remain_neighbour))

        # print(break_1hop_neighbour)
        # print(remain_1hop_neighbour)
        # print(len(list(set(sum(remain_1hop_neighbour, [])))))
        # print(len(remain_1hop_neighbour))

        # store the multihop break neighbour and remain neighbour list for each node
        for node in range(config_num_of_agents):
            break_multihop_neighbour, remain_multihop_neighbour = [], []

            if node in swarm.remain_list:
                break_neighbour = deepcopy(break_1hop_neighbour[node])
                remain_neighbour = deepcopy(remain_1hop_neighbour[node])

                while len(remain_neighbour) > 0:
                    break_multihop_neighbour.append(deepcopy(break_neighbour))
                    remain_multihop_neighbour.append(deepcopy(remain_neighbour))

                    break_neighbour, remain_neighbour = [], []

                    for neighbour in remain_multihop_neighbour[-1]:
                        for n in break_1hop_neighbour[neighbour]:
                            if n not in sum(break_multihop_neighbour, []) and n not in break_neighbour and n != node:
                                break_neighbour.append(n)
                        for n in remain_1hop_neighbour[neighbour]:
                            if n not in sum(remain_multihop_neighbour, []) and n not in remain_neighbour and n != node:
                                remain_neighbour.append(n)

                # print(node, break_multihop_neighbour)
                # print(node, remain_multihop_neighbour)
                # print(node, len(remain_multihop_neighbour), len(sum(remain_multihop_neighbour, [])))

            break_multihop_neighbour.append(deepcopy(break_neighbour))
            remain_multihop_neighbour.append(deepcopy(remain_neighbour))
            
            break_neighbour_list.append(deepcopy(break_multihop_neighbour))
            remain_neighbour_list.append(deepcopy(remain_multihop_neighbour))

        # print(break_neighbour_list)
        # print(remain_neighbour_list)

    print("=======================================")
    print(f"step {step+1} / {max_step} :")
 
    remain_positions_detect = [[], []]
    remain_positions_undetect = [[], []]

    # if step < 24:
    #     continue
    
    for node in swarm.remain_list:
        detect_break_flag = False
        node_remain_list = deepcopy(sum(neighbour_list[node], []))

        maxhop = step+1 if step+1 < len(remain_neighbour_list) else len(remain_neighbour_list)
        for n in sum(break_neighbour_list[node][:maxhop], []):
            node_remain_list.remove(n)

        # print(neighbour_list[node])

        remain_local_positions = [environment_positions[node].tolist()]
        for i in node_remain_list:
            remain_local_positions.append(environment_positions[i].tolist())

        remain_local_positions = np.array(remain_local_positions)

        A = Utils.make_A_matrix(remain_local_positions, len(remain_local_positions), config_communication_range)
        D = Utils.make_D_matrix(A, len(remain_local_positions))
        L = D - A
        connected_flag, num_of_clusters = Utils.check_number_of_clusters(L, len(remain_local_positions))
            
        # print(f"node {node} has {len(remain_local_positions)} neighbour reamained, detect cluster num : {num_of_clusters}")

        if num_of_clusters > 1:
            detect_break_flag = True
            detect_num += 1
            # print("break is detected!")
            remain_positions_detect[0].append(environment_positions[node][0])
            remain_positions_detect[1].append(environment_positions[node][1])
        else:
            remain_positions_undetect[0].append(environment_positions[node][0])
            remain_positions_undetect[1].append(environment_positions[node][1])
            
            # if step == 24:
            #     plt.clf()
            #     plt.title(f"node {node} with {len(node_remain_list)} neighbour")
            #     remain_positions = [[environment_positions[n][0] for n in node_remain_list], [environment_positions[n][1] for n in node_remain_list]]
            #     plt.scatter(remain_positions[0], remain_positions[1], c='green', marker='o', s=8)
            #     remain_positions = [[environment_positions[n][0] for n in sum(remain_neighbour_list[node], [])], [environment_positions[n][1] for n in sum(remain_neighbour_list[node], [])]]
            #     plt.scatter(remain_positions[0], remain_positions[1], c='red', marker='o', s=8)
            #     plt.scatter(environment_positions[node][0], environment_positions[node][1], c='blue', marker='o', s=10)
            #     plt.savefig(f'./Figs/node{node}.png')

            #     print(node, len(remain_neighbour_list[node]), len(sum(remain_neighbour_list[node], [])), len(node_remain_list), num_of_clusters, detect_break_flag)
            #     print(node, remain_neighbour_list[node], neighbour_list[node], break_neighbour_list[node], break_1hop_neighbour[node])
            #     print(node_remain_list)
            #     print()
        
    print(f"{detect_num} of {len(swarm.remain_list)} detect the break\n")
    storage_detection_states.append(detect_break_flag)

    # plt.clf()
    # plt.title(f"step{step+1}: {detect_num}/{len(swarm.remain_list)} detect")
    # plt.scatter(break_positions[0], break_positions[1], c='black', marker='x', s=2)
    # plt.scatter(remain_positions_detect[0], remain_positions_detect[1], c='red', marker='o', s=8)
    # plt.scatter(remain_positions_undetect[0], remain_positions_undetect[1], c='green', marker='o', s=8)
    # plt.savefig(f'./Figs/step{step+1}.png')
    # # plt.show()

        




