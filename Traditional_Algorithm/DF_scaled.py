from copy import deepcopy
import numpy as np

alpha = 10

# differential feilds
def df_scaled(node_global_positions, remain_list, dimension, d, khop=4):
    """
    fly to the center of the swarm is thinks
    :param node_global_positions:
    :return:
    """
    speed = []
    remain_positions = []
    damage_positions = []

    damage_list = [i for i in range(len(node_global_positions)) if i not in remain_list]
    
    for i in remain_list:
        remain_positions.append(deepcopy(node_global_positions[i]))
    for i in damage_list:
        damage_positions.append(deepcopy(node_global_positions[i]))

    remain_positions = np.array(remain_positions)
    damage_positions = np.array(damage_positions)

    # final_positions = np.mean(remain_positions, 0)
    # final_positions = np.array([500, 500]) if dimension == 2 else np.array([500, 500, 50])
    final_positions = np.mean(damage_positions, axis=0)

    all_neighbour = calculate_khop_neighbour(node_global_positions, d)

    for i in range(len(node_global_positions)):
        if i not in remain_list:
            speed.append([0 for _ in range(dimension)])
            continue

        self_positions = node_global_positions[i]
        
        damage_neighbour_directions = []
        for k in range(khop):
            if k <= len(all_neighbour[i]):
                for node in all_neighbour[i][k]:
                    damage_neighbour_directions.append(node_global_positions[node] - self_positions)

        damage_neighbour_directions = np.array(damage_neighbour_directions)
        
        flying_direction = alpha * np.mean(damage_neighbour_directions, axis=0) + (final_positions - self_positions)
        # print(np.linalg.norm(np.mean(damage_neighbour_directions, axis=0)), np.linalg.norm(final_positions - self_positions))
        speed.append(flying_direction)

    speed = np.array(speed)
    max_dis = np.max(np.linalg.norm(speed, axis=1))
    speed = np.array(speed / max_dis)
    # print(len(speed))
    return deepcopy(speed)

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