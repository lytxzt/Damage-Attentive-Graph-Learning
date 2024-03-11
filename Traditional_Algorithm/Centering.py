from copy import deepcopy
import numpy as np


def centering_fly(node_global_positions, remain_list, index):
    """
    fly to the center of the swarm is thinks
    :param node_global_positions:
    :return:
    """
    remain_positions = []
    self_positions = node_global_positions[index]
    for i in remain_list:
        remain_positions.append(deepcopy(node_global_positions[i]))
    remain_positions = np.array(remain_positions)
    final_positions = np.mean(remain_positions, 0)
    flying_direction = (final_positions - self_positions)/np.linalg.norm(final_positions - self_positions)
    return deepcopy(flying_direction)

def centering_scaled(node_global_positions, remain_list):
    """
    fly to the center of the swarm is thinks
    :param node_global_positions:
    :return:
    """
    speed = []
    remain_positions = []
    max_dis = 1
    
    for i in remain_list:
        remain_positions.append(deepcopy(node_global_positions[i]))
    remain_positions = np.array(remain_positions)
    final_positions = np.mean(remain_positions, 0)

    for i in range(len(node_global_positions)):
        if i not in remain_list:
            speed.append([0,0,0])
            continue
        self_positions = node_global_positions[i]
        flying_direction = final_positions - self_positions
        speed.append(flying_direction)
        if np.linalg.norm(final_positions - self_positions) > max_dis:
            max_dis = np.linalg.norm(final_positions - self_positions)

    speed = np.array(speed / max_dis)
    # print(len(speed))
    return deepcopy(speed)
