from Environment import Environment
from Swarm import Swarm
from Configurations import *
import Utils
import matplotlib.animation as animation
from Drawing.Draw_Static import *
from Main_algorithm_GCN.GCO import GCO

# determine if draw the video
"""
Note: if true, it may take a little long time
"""
config_draw_video = False

# determine if use meta learning param
"""
Note: if use trained meta param, you need to down the trained meta parameters from 
       https://cloud.tsinghua.edu.cn/f/2cb28934bd9f4bf1bdd7/ or 
       https://drive.google.com/file/d/1QPipenDZi_JctNH3oyHwUXsO7QwNnLOz/view?usp=sharing
       the size of meta parameter file is pretty large (about 1.2GB)
       otherwise, you could run the Meta-learning_all.py file to train the meta parameter on your own machine
"""
meta_param_use = False

"""
    algorithm mode: 0 for CSDS
                    1 for HERO
                    2 for CEN
                    3 for SIDR
                    4 for GCN-2017
                    5 for CR-MGC (proposed algorithm)
"""
# set this value to 5 to run the proposed algorithm
config_algorithm_mode = 3
algorithm_mode = {0: "CSDS",
                  1: "HERO",
                  2: "CEN",
                  3: "SIDR",
                  4: "GCN_2017",
                  5: "CR-MGC (proposed algorithm)"}

print("SCC problem Starts...")
print("------------------------------")
print("Algorithm: %s" % (algorithm_mode[config_algorithm_mode]))

# storage
storage_remain_list = []
storage_positions = []
# storage_destroy_positions = []
storage_connection_states = []
storage_remain_connectivity_matrix = []

# change the number of destructed UAVs
config_num_destructed_UAVs = 100  # should be in the range of [1, config_num_-2]

# change the seed to alternate the UED
np_rand_seed = [61, 29, 83, 3, 59, 22, 8, 96, 80, 20, 39, 19, 89, 75, 79, 55, 61, 74, 8, 89, 83, 3, 38, 88, 56, 68, 67, 46, 48, 63, 54, 43, 52, 72, 75, 21, 64, 44, 50, 77, 39, 14, 18, 66, 82, 51, 65, 90, 57, 35, 92, 74, 9, 64, 52, 91, 56, 87, 77, 82, 34, 38, 38, 97, 3, 85, 67, 15, 86, 21, 67, 68, 98, 99, 32, 100, 48, 19, 85, 67, 81, 25, 64, 23, 37, 87, 31, 8, 96, 47, 63, 57, 1, 66, 18, 89, 54, 60, 58, 25]
rand_seed = [47, 94, 7, 7, 1, 31, 7, 33, 80, 43, 74, 82, 61, 93, 96, 93, 95, 13, 5, 75, 7, 71, 23, 14, 78, 38, 38, 40, 89, 57, 20, 1, 86, 97, 13, 79, 27, 47, 67, 19, 96, 27, 54, 44, 26, 84, 55, 42, 61, 43, 94, 84, 17, 65, 91, 52, 48, 34, 90, 84, 31, 90, 51, 74, 1, 21, 94, 44, 28, 19, 70, 95, 69, 36, 71, 90, 67, 82, 46, 91, 20, 21, 73, 67, 9, 58, 34, 59, 92, 23, 42, 7, 19, 5, 91, 42, 73, 15, 47, 1]
# np_rand_seed = [61, 29, 83, 3, 59, 22, 8, 96, 80, 20]
# rand_seed = [47, 94, 7, 7, 1, 31, 7, 33, 80, 43]
# np.random.seed(17)
# random.seed(18)

storage_random_seed = []
storage_connect_step = []
storage_connect_positons = []

for case in range(40,50):
    environment = Environment()
    if algorithm_mode == 0:
        swarm = Swarm(algorithm_mode=config_algorithm_mode, enable_csds=True, meta_param_use=meta_param_use)
    else:
        swarm = Swarm(algorithm_mode=config_algorithm_mode, enable_csds=False, meta_param_use=meta_param_use)
    num_cluster_list = []

    environment_positions = environment.reset()
    swarm.reset()

    np.random.seed(np_rand_seed[case])
    random.seed(rand_seed[case])

    # destruction
    storage_remain_list.append(deepcopy(swarm.remain_list))
    storage_positions.append(deepcopy(swarm.true_positions))
    # storage_destroy_positions.append([])
    storage_connection_states.append(True)
    storage_remain_connectivity_matrix.append(
        deepcopy(Utils.make_A_matrix(swarm.true_positions, config_num_of_agents, config_communication_range)))

    # flag if break the CCN
    break_CCN_flag = True
    # num of connected steps
    num_connected_steps = 0

    destroy_num, destroy_list = environment.stochastic_destroy(mode=2, num_of_destroyed=config_num_destructed_UAVs)
    swarm.destroy_happens(deepcopy(destroy_list), deepcopy(environment_positions))

    print("=======================================")

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
        print("case %d: not distructed" % (case))
        continue

    positions_with_clusters = Utils.split_the_positions_into_clusters(initial_remain_positions, num_cluster, A)

    for step in range(450):
        actions, max_time = swarm.take_actions()
        environment_next_positions = environment.next_state(deepcopy(actions))
        swarm.update_true_positions(environment_next_positions)

        temp_cluster = environment.check_the_clusters()
        num_cluster_list.append(temp_cluster)
        # print("---------------------------------------")
        if temp_cluster == 1:
            print("case %d: step %d ---num of clusters %d -- connected" % (case, step, environment.check_the_clusters()))
        else:
            num_connected_steps += 1
            print("case %d: step %d ---num of clusters %d -- unconnected" % (case, step, environment.check_the_clusters()), end='\r')
            if step == 449:
                print("case %d: step %d ---num of clusters %d -- unconnected" % (case, step, environment.check_the_clusters()))

        storage_remain_list.append(deepcopy(swarm.remain_list))

        storage_positions.append(deepcopy(environment_next_positions))
        
        if environment.check_the_clusters() == 1:
            storage_connection_states.append(True)
        else:
            storage_connection_states.append(False)

        remain_positions = []
        for i in swarm.remain_list:
            remain_positions.append(deepcopy(environment_next_positions[i]))
        remain_positions = np.array(remain_positions)
        storage_remain_connectivity_matrix.append(
            deepcopy(Utils.make_A_matrix(remain_positions, len(swarm.remain_list), config_communication_range)))

        if environment.check_the_clusters() == 1:
            # print("  case %d, step %f" % (case, step))
            break

        # update
        environment.update()
        environment_positions = deepcopy(environment_next_positions)

    # print("case %d: step %d ---num of clusters %d -- connected" % (case, step, environment.check_the_clusters()))
    
    final_positions = []
    for i in swarm.remain_list:
        final_positions.append(deepcopy(storage_positions[-1][i]))

    storage_random_seed.append(case)
    storage_connect_step.append(step)
    storage_connect_positons.append(final_positions)

print(storage_random_seed)
print(storage_connect_positons)
print(storage_connect_step)
        