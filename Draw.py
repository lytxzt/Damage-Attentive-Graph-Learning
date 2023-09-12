import Utils
from copy import deepcopy
from Configurations import *
from Environment import Environment
from Swarm import Swarm
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

np.random.seed(57)
random.seed(77)

environment = Environment()
swarm = Swarm()

environment_positions = environment.reset()
destroy_num, destroy_list = environment.stochastic_destroy(mode=2, num_of_destroyed=100)

swarm.destroy_happens(deepcopy(destroy_list), deepcopy(environment_positions))

# print(len(swarm.remain_list))

x = [swarm.remain_positions[i][1] for i in swarm.remain_list]
y = [swarm.remain_positions[i][0] for i in swarm.remain_list]

z = []
for i in swarm.remain_list:
    cnt = 0
    for j in swarm.remain_list:
        if i == j: continue

        if np.sqrt(np.sum((swarm.remain_positions[i] - swarm.remain_positions[j])**2)) <= 120:
            cnt += 1

    z.append(cnt)

# print(z)
x = np.array(x)
y = np.array(y)
z = np.array(z)

xlist = np.linspace(0, 1000, 1000)
ylist = np.linspace(0, 1000, 1000)
xlist, ylist = np.meshgrid(xlist, ylist)

zdata = griddata((x, y), z, (xlist, ylist), method='cubic')
# zdata = np.sqrt(xlist**2+ylist**2)

# fig = plt.figure()
# ax = Axes3D(fig)
# fig.add_axes(ax)
# ax.plot_surface(xlist, ylist, zdata, cmap='jet', vmin=0, vmax=10)
# ax.scatter(x, y, z, s=30, c='g')
# # plt.draw()
# plt.show()

# sns.heatmap(zdata, cmap="jet")
ax=plt.gca()
plt.imshow(zdata, cmap='jet', vmin=0, vmax=10)
plt.scatter(x, y, c='g', s=30)
ax.invert_yaxis()
plt.show()