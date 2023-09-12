import Utils
from copy import deepcopy
from Configurations import *
from Environment import Environment


class SwarmDelay:
    def __init__(self):
        self.initial_positions = deepcopy(config_initial_swarm_positions)
        self.remain_list = [i for i in range(config_num_of_agents)]
        self.remain_num = config_num_of_agents
        self.num_of_agents = config_num_of_agents
        self.max_destroy_num = config_maximum_destroy_num

        self.remain_positions = deepcopy(self.initial_positions)
        self.true_positions = deepcopy(self.initial_positions)

        self.database = [{"known_positions": deepcopy(self.initial_positions),
                          "existing_list": [i for i in range(config_num_of_agents)],
                          "connected": True,
                          "if_destroyed": False} for i in range(config_num_of_agents)]
        
        self.all_neighbour = []
        self.maxhop = 1

    def destroy_happens(self, destroy_list, environment_positions):
        self.notice_destroy = True
        for destroy_index in destroy_list:
            self.remain_list.remove(destroy_index)
        self.true_positions = deepcopy(environment_positions)
        self.remain_num = len(self.remain_list)

        self.make_remain_positions()

    def make_remain_positions(self):
        self.remain_positions = []
        for i in self.remain_list:
            self.remain_positions.append(deepcopy(self.true_positions[i]))
        self.remain_positions = np.array(self.remain_positions)

    def calculate_neighbour(self):
        neighbour_i_1hop = []
        for i in range(self.num_of_agents):
            neighbour = []
            for j in range(self.num_of_agents):
                if i==j: continue

                if euclidean(self.initial_positions[i], self.initial_positions[j]) <= config_communication_range:
                    neighbour.append(j)

            neighbour_i_1hop.append(deepcopy(neighbour))

        # print(neighbour_i_1hop)

        self.all_neighbour = []
        for i in range(self.num_of_agents):
            cnt = 0
            neighbour_i_all = []
            neighbour_i_multihop = deepcopy(neighbour_i_1hop[i])
            # neighbour_i_all.append(deepcopy(neighbour_i_1hop[i]))

            # while len(sum(neighbour_i_all, [])) < 199:
            while len(neighbour_i_multihop) > 0:
                # print(len(sum(neighbour_i_all, [])))
                # print(neighbour_i_all)
                neighbour_i_all.append(deepcopy(neighbour_i_multihop))
                neighbour_i_multihop = []

                for j in neighbour_i_all[cnt]:
                    for k in neighbour_i_1hop[j]:
                        if k not in sum(neighbour_i_all, []) and k not in neighbour_i_multihop and k != i:
                            neighbour_i_multihop.append(k)

                # neighbour_i_all.append(deepcopy(neighbour_i_multihop))
                cnt += 1
                
            # print(neighbour_i_all)
            self.all_neighbour.append(deepcopy(neighbour_i_all))

        # print(self.all_neighbour)
        for neighbour in self.all_neighbour: 
            # print(neighbour)
            if len(neighbour) > self.maxhop:
                self.maxhop = len(neighbour)

        print("max hop is %d" % self.maxhop)
        
    # def destory_detection(self):
    #     for step in range(100):
    #         if step == 0:
    #             self.calculate_neighbour()
            


def euclidean(x, y):
    return np.sqrt(np.sum((x - y)**2))


if __name__ == "__main__":
    np.random.seed(17)
    random.seed(18)

    environment = Environment()
    swarm = SwarmDelay()

    # environment_positions = environment.reset()
    # destroy_num, destroy_list = environment.stochastic_destroy(mode=2, num_of_destroyed=100)

    # swarm.destroy_happens(deepcopy(destroy_list), deepcopy(environment_positions))

    # print(swarm.remain_list)
    swarm.calculate_neighbour()
    # l = [[47, 64, 87, 93, 155], [0, 43, 68, 117, 192, 88, 107, 151, 181, 92], [18, 15, 111, 119, 182, 66], [71, 85, 142, 5, 6, 54, 25, 42, 74], [145, 197, 102, 137], [29, 158, 89, 140, 152, 51], [76, 114, 127, 67, 31, 101, 57, 60, 81], [40, 128, 178, 171, 12, 21, 24], [44, 161, 135, 122, 126, 136, 185], [52, 83, 157, 9, 17, 175, 176, 59, 84, 141], [16, 27, 132, 35, 50, 146, 164, 195, 10, 130, 133, 169, 173], [2, 69, 105, 179, 113, 75, 13, 39, 110, 123], [104, 109, 149, 183, 33, 65, 184, 28, 37, 19, 168, 98, 143, 159], [124, 191, 63, 100, 4, 46, 90, 121, 125, 167, 99, 72, 165, 106], [45, 73, 156, 190, 32, 77, 144, 180, 23, 55, 56, 174, 199, 8, 196], [36, 153, 186, 188, 96, 103, 120, 134, 177, 14, 26, 94, 139, 147, 166, 189, 148, 61, 38, 112, 172, 30, 138], [194, 49, 70, 198, 154, 79, 97, 131, 7, 62, 108, 53, 150, 20, 82, 34, 78, 80], [129, 115, 48, 58], [160, 170, 193]]
    # print(len(l), len(sum(l, [])))

    # l2 = [[47, 64, 87, 93, 155], [22, 86, 163], [16, 27, 104, 105, 109, 149, 179, 183], [41, 162], [19, 46, 90, 121, 125, 167, 199], [85, 111, 142], [119, 145, 197], [14, 20, 26, 62, 79, 94, 97, 120, 131, 139, 147, 148], [30, 99, 138], [35, 135], [13, 39, 50, 130, 164, 175], [187], [24, 81], [10, 19, 39, 168], [7, 20, 26, 62, 77, 79, 82, 94, 97, 120, 131, 139, 147, 148, 166, 189], [68, 85, 111, 142], [2, 27, 52, 69, 105, 132], [50, 122, 146, 164, 175, 176, 195], [43, 71, 117, 182], [4, 13, 46, 90, 121, 125, 167, 168], [7, 14, 26, 62, 82, 94, 131, 147, 166], [24, 81, 126, 136, 185], [1, 116, 163], [55, 56, 61, 63, 77, 139, 147, 166, 174, 189], [12, 21, 81, 185], [42, 66], [7, 14, 20, 62, 77, 79, 94, 97, 120, 131, 139, 147, 148, 166], [2, 16, 52, 105, 132, 179, 195], [109, 124, 149, 179, 183], [158, 197], [8, 78, 80, 196], [101, 140, 152], [96, 103, 120, 134, 144, 177, 180, 191], [69, 104, 105, 184], [38], [9, 113], [45, 73, 153, 190, 199], [75, 98, 100, 110], [34, 56], [10, 13], [44, 76, 127, 161, 171, 178], [3, 162], [25, 66, 74, 137], [18, 47, 93, 117, 192], [40, 52, 122, 161, 178], [36, 73, 124, 153, 156, 186, 188, 190], [4, 19, 90, 121, 125, 167, 168], [0, 43, 64, 68, 93, 117, 192], [53, 150, 160, 170], [70, 129, 186, 188, 194, 198], [10, 17, 130, 133, 164, 175, 195], [57, 60, 81, 137], [16, 27, 44, 122, 132], [48, 150, 177], [71, 102, 182], [23, 61, 63, 77, 139, 147, 166, 174, 189], [23, 38, 63, 174], [51, 60], [150, 193], [84, 126, 136, 141, 185], [51, 57], [23, 55, 112, 147, 166, 172, 174], [7, 14, 20, 26, 79, 82, 94, 97, 120, 131, 139, 147, 148], [23, 55, 56, 174, 184], [0, 47, 68, 87, 88, 107, 151, 155, 181], [69, 184, 191], [25, 42, 74, 88], [89], [15, 47, 64, 107, 111, 151, 155, 181], [16, 33, 65, 104, 105, 183, 184], [49, 129, 186, 188, 194, 198], [18, 54, 182], [98, 165, 196], [36, 45, 124, 153, 156, 186, 190], [42, 66, 137], [37, 110, 130, 133, 164], [40, 127, 128, 158, 178], [14, 23, 26, 55, 94, 139, 147, 166, 189, 191], [30, 80], [7, 14, 26, 62, 94, 97, 103, 120, 131, 139, 148, 150], [30, 78], [12, 21, 24, 51], [14, 20, 62, 139, 147, 166], [132, 157, 161], [59, 126, 136, 141, 169, 173, 185], [5, 15, 111, 142], [1, 163], [0, 64, 93, 155], [64, 66, 107, 151, 155, 181], [67, 102], [4, 19, 46, 121, 125, 167, 168], [163], [151, 155], [0, 43, 47, 87, 117], [7, 14, 20, 26, 62, 77, 79, 97, 120, 131, 139, 147, 148, 166], [118], [32, 103, 134, 144, 154, 177, 180], [7, 14, 26, 62, 79, 94, 103, 120, 131, 148, 150, 177], [37, 72, 100, 110, 165], [8, 168], [37, 98, 143, 159], [31, 140], [54, 89, 140, 152], [32, 79, 96, 97, 120, 131, 144, 148, 154, 177, 180], [2, 33, 69, 105, 149, 183, 184], [2, 16, 27, 33, 69, 104, 132, 179, 183], [143, 165], [64, 68, 88, 151, 155, 181], [134], [2, 28, 124, 149, 179, 183], [37, 75, 98, 130, 133], [5, 15, 68, 85, 142], [61, 172, 174], [35], [158, 171], [154], [22], [18, 43, 47, 93, 119, 182, 192], [95], [6, 117, 182, 192], [7, 14, 26, 32, 62, 79, 94, 97, 103, 131, 144, 148, 177, 180], [4, 19, 46, 90, 125, 167, 168], [17, 44, 52, 175, 176, 178], [143, 159, 169, 173], [28, 45, 73, 109, 149, 156, 190], [4, 19, 46, 90, 121, 167, 199], [21, 59, 84, 136, 141, 185], [40, 76, 128, 158, 178], [76, 127, 135], [49, 70, 198], [10, 50, 75, 110, 133, 164, 175], [7, 14, 20, 26, 62, 79, 94, 97, 103, 120, 139, 148], [16, 27, 52, 83, 105, 157], [50, 75, 110, 130, 164, 175], [32, 96, 108], [9, 128], [21, 59, 84, 126, 185], [42, 51, 74], [8], [7, 14, 23, 26, 55, 62, 77, 79, 82, 94, 131, 147, 166, 189], [31, 101, 102, 152], [59, 84, 126, 169, 173], [5, 15, 85, 111], [100, 106, 123, 159, 165], [32, 96, 103, 120, 148, 177, 180, 191], [6, 197], [17, 175, 176], [7, 14, 20, 23, 26, 55, 61, 62, 77, 82, 94, 139, 166, 189], [7, 14, 26, 62, 79, 94, 97, 103, 120, 131, 144, 177, 180], [2, 28, 104, 109, 124, 179, 183], [48, 53, 58, 79, 97, 177], [64, 68, 88, 92, 107, 155, 181], [31, 102, 140], [36, 45, 73, 186, 188, 194], [96, 103, 115, 177], [0, 64, 68, 87, 88, 92, 107, 151, 181], [45, 73, 124], [83, 132, 161], [29, 76, 114, 127, 197], [100, 123, 143, 165], [48, 170, 193], [40, 44, 83, 157], [3, 41], [1, 22, 86, 91], [10, 17, 50, 75, 130, 133, 175, 195], [72, 98, 106, 143, 159], [14, 20, 23, 26, 55, 61, 77, 82, 94, 139, 147, 189], [4, 19, 46, 90, 121, 125], [13, 19, 46, 90, 99, 121], [84, 123, 141, 173], [48, 160, 193], [40, 114], [61, 112, 174], [84, 123, 141, 169], [23, 55, 56, 61, 63, 112, 172], [10, 17, 50, 122, 130, 133, 146, 164, 176, 195], [17, 122, 146, 175], [32, 53, 96, 97, 103, 120, 144, 148, 150, 154, 180], [40, 44, 76, 122, 127], [2, 27, 28, 105, 109, 149, 183], [32, 96, 103, 120, 144, 148, 177, 191], [64, 68, 88, 107, 151, 155], [18, 54, 71, 117, 119, 192], [2, 28, 69, 104, 105, 109, 149, 179, 184], [33, 63, 65, 69, 104, 183], [21, 24, 59, 84, 126, 136], [45, 49, 70, 73, 153, 188, 194, 198], [11], [45, 49, 70, 153, 186, 194], [14, 23, 55, 77, 139, 147, 166], [36, 45, 73, 124, 199], [32, 65, 77, 144, 180], [43, 47, 117, 119, 182], [58, 160, 170], [49, 70, 153, 186, 188, 198], [17, 27, 50, 164, 175], [30, 72], [6, 29, 145, 158], [49, 70, 129, 186, 194], [4, 36, 125, 190]]

    # for i in range(200):
    #     if i not in sum(l, []):
    #         print(i)


