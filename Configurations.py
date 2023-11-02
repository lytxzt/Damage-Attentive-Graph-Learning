import pandas as pd
import numpy as np
import random
import torch

"""
specify a certain GPU
"""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'

# random seed
np.random.seed(1)
random.seed(2)
torch.manual_seed(1)

config_dimension = 2
config_initial_swarm_positions = pd.read_excel("Configurations/swarm_positions_200.xlsx")
config_initial_swarm_positions = config_initial_swarm_positions.values[:, 1:1+config_dimension]
config_initial_swarm_positions = np.array(config_initial_swarm_positions, dtype=np.float64)

# configurations on swarm
config_num_of_agents = 200
config_communication_range = 120

# configurations on environment
config_width = 1000.0
config_length = 1000.0
config_height = 100.0

config_constant_speed = 1

# configurations on destroy
config_maximum_destroy_num = 50
config_minimum_remain_num = 5

# configurations on meta learning
config_meta_training_epi = 500
# configurations on Graph Convolutional Network
config_K = 1 / 100
config_best_eta = 0.3
config_best_epsilon = 0.99

# configurations on one-off UEDs
config_num_destructed_UAVs = 100  # should be in the range of [1, num_of_UAVs-2]
config_normalize_positions = True

# configurations on training GCN
config_alpha_k = [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.9, 0.95, 1, 1.5, 2, 3, 5]
config_gcn_repeat = 100
config_expension_alpha = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
config_d0_alpha = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

config_representation_step = 450

