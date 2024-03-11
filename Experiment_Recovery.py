from Environment import Environment
from Swarm import Swarm
from Configurations import *
import Utils
import matplotlib.animation as animation
from copy import deepcopy
from Main_algorithm_GCN.GCO import GCO
import matplotlib.pyplot as plt

# determine if draw the video
"""
Note: if true, it may take a little long time
"""
config_draw_video = True
show_degree = True

# determine if use meta learning param
meta_param_use = False

"""
    algorithm mode: 0 for CSDS
                    1 for HERO
                    2 for CEN
                    3 for SIDR
                    4 for GCN-2017
                    5 for CR-MGC
                    6 for DEMD (proposed algorithm)
                    7 for DD-GCN (imcomplete)
"""
# set this value to 6 to run the proposed algorithm
config_algorithm_mode = 6
algorithm_mode = {0: "CSDS",
                  1: "HERO",
                  2: "CEN",
                  3: "SIDR",
                  4: "GCN_2017",
                  5: "CR-MGC",
                  6: "DEMD",
                  7: "DD-GCN"}

print("SCC problem Starts...")
print("------------------------------")
print("Algorithm: %s" % (algorithm_mode[config_algorithm_mode]))

environment = Environment()
if algorithm_mode == 0:
    swarm = Swarm(algorithm_mode=config_algorithm_mode, enable_csds=True, meta_param_use=meta_param_use)
else:
    swarm = Swarm(algorithm_mode=config_algorithm_mode, enable_csds=False, meta_param_use=meta_param_use)
num_cluster_list = []

environment_positions = environment.reset()
swarm.reset()

# storage
storage_remain_list = []
storage_positions = []
# storage_destroy_positions = []
storage_connection_states = []
storage_remain_connectivity_matrix = []

# change the number of destructed UAVs
config_num_destructed_UAVs = 100  # should be in the range of [1, config_num_-2]

# change the seed to alternate the UED
seed = 39
np.random.seed(17)
random.seed(18)
# np.random.seed(seed)
# random.seed(seed)

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
for step in range(150):
    # destroy at time step 0
    if step == 0:
        print("=======================================")
        print("destroy %d -- mode %d num %d " % (0, 2, config_num_destructed_UAVs))
        destroy_num, destroy_list = environment.stochastic_destroy(mode=2, num_of_destroyed=config_num_destructed_UAVs)
        # real_destroy_list = [64, 83, 50, 61, 49, 29, 74, 121, 142, 156, 60, 37, 191, 161, 169, 128, 0, 71, 7, 127, 44, 48, 129, 99, 46, 84, 82, 30, 22, 141, 66, 136, 120, 52, 57, 195, 18, 20, 51, 143, 112, 152, 137, 19, 25, 54, 174, 171, 24, 147]
        # destroy_num, destroy_list = environment.stochastic_destroy(mode=4, real_destroy_list=real_destroy_list)
        print("destroy %d nodes \ndestroy index list :" % config_num_destructed_UAVs)
        print(destroy_list)
        swarm.destroy_happens(deepcopy(destroy_list), deepcopy(environment_positions))

        # destroy_list_panda = pd.DataFrame(np.array(destroy_list))
        # Utils.store_dataframe_to_excel(destroy_list_panda, "Experiment_Fig/one_off_UEDs/destroy_list.xlsx")

        # draw the destroy picture, i.e., Fig. 12(a)
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
        # check if the UED break the CCN of the USNET
        if num_cluster == 1:
            print("---------------------------------------")
            print("This kind of UED does not break the CCN of the USNET!")
            print("Please change the random seed or the number of destructed UAVs!")
            print("Algorithm Ends")
            print("---------------------------------------")
            break_CCN_flag = False
            break
        positions_with_clusters = Utils.split_the_positions_into_clusters(initial_remain_positions,
                                                                          num_cluster, A)
        # draw_pic_with_destroyed(config_num_of_agents - config_num_destructed_UAVs, config_num_destructed_UAVs,
        #                         initial_remain_positions, positions_with_clusters,
        #                         destroy_positions, num_cluster, A)

        # draw gco iteration i.e., Fig. 12(b)
        print("=======================================")
        print("Applying graph convolutional operations...")
        # gco = GCO()
        # _, __, ___, storage_gco_positions = gco.gco(config_initial_swarm_positions, swarm.remain_list)
        # draw_approximate_pic_2D(config_num_of_agents - config_num_destructed_UAVs, storage_gco_positions)

    actions, max_time = swarm.take_actions()
    environment_next_positions = environment.next_state(deepcopy(actions))
    swarm.update_true_positions(environment_next_positions)

    temp_cluster = environment.check_the_clusters()
    num_cluster_list.append(temp_cluster)
    print("---------------------------------------")
    if temp_cluster == 1:
        print(f"step {step} ---num of clusters {environment.check_the_clusters()} -- connected")
    else:
        num_connected_steps += 1
        print(f"step {step} ---num of clusters {environment.check_the_clusters()} -- disconnected --max time {max_time}")

    storage_remain_list.append(deepcopy(swarm.remain_list))

    storage_positions.append(deepcopy(environment_next_positions))
    # if step >= 10:
    #     temp_positions = []
    #     for i in destroy_list:
    #         temp_positions.append(deepcopy(environment_next_positions[i]))
    #     storage_destroy_positions.append(deepcopy(temp_positions))
    # else:
    #     storage_destroy_positions.append([])
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

    # if environment.check_the_clusters() == 1:
    #     break

    # update
    environment.update()
    environment_positions = deepcopy(environment_next_positions)


if break_CCN_flag:
    # draw trajectory pic i.e., Fig. 12(c)
    print("=======================================")
    print("plotting trajectories of nodes...")
    final_positions = []
    for i in swarm.remain_list:
        final_positions.append(deepcopy(storage_positions[-1][i]))
    final_positions = np.array(final_positions)
    # draw_once_two_nodes(config_num_of_agents - config_num_destructed_UAVs, initial_remain_positions, final_positions)
    # print(final_positions)


    def update(frame):
        # ax = Axes3D(fig)
        plt.clf()
        ax = plt.axes(xlim=(0, 1000), ylim=(0, 1000))

        for i in range(len(storage_remain_list[frame])):
            for j in range(i, len(storage_remain_list[frame])):
                if storage_remain_connectivity_matrix[frame][i, j] == 1:
                    x = [storage_positions[frame][storage_remain_list[frame][i], 0],
                         storage_positions[frame][storage_remain_list[frame][j], 0]]
                    y = [storage_positions[frame][storage_remain_list[frame][i], 1],
                         storage_positions[frame][storage_remain_list[frame][j], 1]]
                    # z = [storage_positions[frame][storage_remain_list[frame][i], 2],
                    #      storage_positions[frame][storage_remain_list[frame][j], 2]]
                    ax.plot(x, y, c='lightsteelblue', zorder=1)

        for i in range(config_num_of_agents):
            if i in storage_remain_list[frame]:
                x = [storage_positions[frame][i, 0],
                     config_initial_swarm_positions[i, 0]]
                y = [storage_positions[frame][i, 1],
                     config_initial_swarm_positions[i, 1]]
                # z = [storage_positions[frame][i, 2],
                #      config_initial_swarm_positions[i, 2]]
                ax.plot(x, y, c='blue', zorder=2)
                
        for i in range(config_num_of_agents):
            if i in storage_remain_list[frame]:
                ax.scatter(storage_positions[frame][i, 0], storage_positions[frame][i, 1],
                        #    storage_positions[frame][i, 2],
                           s=30, c='g', zorder=4)
            else:
                if frame <= 10:
                    # red = int(50 - (50 - 10) * (frame / trajectory_step))
                    # green = int(230 - (230 - 30) * (i / trajectory_step))
                    # blue = int(50 - (50 - 10) * (i / trajectory_step))
                    # c = str(red) + ',' + str(green) + ',' + str(blue)
                    # c = RGB_to_Hex(c)
                    ax.scatter(storage_positions[frame][i, 0], storage_positions[frame][i, 1],
                            #    storage_positions[frame][i, 2],
                               s=30, c='r', zorder=3)
                # ax.text(storage_positions[frame][i, 0] + 1, storage_positions[frame][i, 1] + 1,
                #         storage_positions[frame][i, 2] + 1,
                #         'Destroyed', c='r')
        # ax.text(5, 5, 0, 'CLEC = %f' % 120, c='blue')
        ax.text(5, 100, 'time steps = %d' % (frame), c='b', zorder=5)
        ax.text(5, 60, 'number_of_clusters = %d' % num_cluster_list[frame], c='b', zorder=5)
        # if frame >= 11:
        #     plt.text(5, 5, -45, 'destroy %d UAVs randomly... ' % config_num_destructed_UAVs, c='r')

        if storage_connection_states[frame]:
            ax.text(5, 20, 'Connected...', c='g')
        else:
            ax.text(5, 20, 'Unconnected...', c='r')
        # plt.set_zlabel('Height', fontdict={'size': 15, 'color': 'black'})
        # ax.ylabel('Ground Y', fontdict={'size': 15, 'color': 'black'})
        # ax.xlabel('Ground X', fontdict={'size': 15, 'color': 'black'})

        # ax.xlim(0, config_width)
        # ax.ylim(0, config_length)
        # ax.set_zlim(-50, 150)
        print("finish frame %d ..." % frame)


    if config_draw_video:
        print("=======================================")
        print("Plotting the dynamic trajectory...")
        fig = plt.figure()
        frame = np.linspace(0, num_connected_steps + 10, num_connected_steps + 11).astype(int)
        ani = animation.FuncAnimation(fig, update, frames=frame, interval=90, repeat_delay=10)
        ani.save("Figs/one_off_destruct_%d.gif" % config_num_destructed_UAVs, writer='pillow', bitrate=2048, dpi=500)
        plt.show()

    # environment.make_remain_positions()
    # store_best_final_positions = pd.DataFrame(environment.remain_positions)
    # Utils.store_dataframe_to_excel(store_best_final_positions, "Experiment_Fig/Experiment_3_compare_GI/positions_1_2.xlsx")
    #
    # storage_remain_list = pd.DataFrame(np.array(storage_remain_list))
    # Utils.store_dataframe_to_excel(storage_remain_list, "video/storage_remain_list.xlsx")
    #

    
    #
    # connection_steps_list_ = pd.DataFrame(np.array(num_cluster_list))
    # if config_algorithm_mode == 6:
    #     Utils.store_dataframe_to_excel(connection_steps_list_,
    #                                    "Experiment_Fig/Experiment_6/CR_GCM_N_num_of_cluster_once_destroy_100.xlsx",
    #                                    sheetname="CR_GCM_N")
        
    if show_degree:
        A = Utils.make_A_matrix(final_positions, len(final_positions), config_communication_range)
        degrees = [int(d) for d in np.sum(A, axis=0)]
        # print(degrees)

        # degrees = [15, 2, 2, 3, 3, 11, 2, 3, 4, 4, 14, 6, 14, 4, 9, 4, 11, 3, 14, 9, 11, 2, 4, 4, 5, 10, 5, 2, 4, 4, 6, 4, 1, 3, 2, 9, 15, 2, 4, 8, 2, 15, 3, 1, 2, 3, 2, 1, 10, 3, 9, 4, 4, 15, 5, 4, 14, 9, 4, 5, 6, 4, 2, 4, 4, 9, 3, 6, 16, 3, 15, 3, 4, 4, 13, 10, 6, 2, 9, 15, 4, 14, 6, 11, 3, 2, 9, 9, 10, 12, 13, 5, 5, 11, 5, 1, 15, 15, 11, 2]
        # degrees = list(np.array(degrees))

        # degree_range = [i+1 for i in range(max(degrees))]
        # # print(degree_range)
        # degree_distribution = np.zeros(len(degree_range))
        # for i in degrees:
        #     degree_distribution[i-1] += 1
        # degree_distribution = list(degree_distribution/100)


        degree_sequence = sorted(degrees, reverse=True)

        plt.figure(figsize=(8, 6))
        plt.hist(degree_sequence, bins=max(degrees), range=[0.5,max(degrees)+0.5], rwidth=0.6)
        # plt.plot(degree_range, degree_distribution)
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Node Num")
        plt.show()
