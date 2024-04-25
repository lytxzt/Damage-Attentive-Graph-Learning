import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg


def draw_khop_figs():
    colors = list(mcolors.TABLEAU_COLORS.keys())

    num_destroyed = [25, 50, 75, 100, 125, 150, 175]

    step_methods = []
    marklist = ['o', '^', 's', 'd', 'h', 'x', 'v']
    methods = [f'$k={i+2}$' for i in range(7)]

    for k in range(7):
        step_k = []
        for dnum in num_destroyed:
            with open(f'./Logs/d{dnum}/DEMD_d{dnum}_k{k+2}.txt', 'r') as f:
                data = f.read()

            data = data.split('\n')
            # print(data[7].replace(' ',''))
            step_k.append(float(data[7].replace(' ','')))

        step_methods.append(step_k)


    plt.figure()
    for i, s in enumerate(step_methods):
        plt.plot(num_destroyed, s, c=mcolors.TABLEAU_COLORS[colors[i]],marker=marklist[i],label=methods[i])

    plt.xlim(10,200)
    plt.ylim(0,540)
    plt.grid(linestyle='--')
    plt.text(20,20, '$k^*$=3', c='firebrick', fontsize=12)
    plt.text(45,52, '$k^*$=4', c='firebrick', fontsize=12)
    plt.text(70,115, '$k^*$=4', c='firebrick', fontsize=12)
    plt.text(95,170, '$k^*$=5', c='firebrick', fontsize=12)
    plt.text(120,245, '$k^*$=5', c='firebrick', fontsize=12)
    plt.text(145,275, '$k^*$=6', c='firebrick', fontsize=12)
    plt.text(170,325, '$k^*$=6', c='firebrick', fontsize=12)
    plt.xlabel('Number of destroyed UAVs', fontdict={'family':'serif', 'size':14})
    plt.ylabel('Average recovery steps $J_S$', fontdict={'family':'serif', 'size':14})
    plt.legend(loc='upper left')
    plt.show()

def draw_method_figs():
    colors = list(mcolors.TABLEAU_COLORS.keys())

    num_destroyed = [25, 50, 75, 100, 125, 150, 175]

    step_methods = []
    marklist = ['o', '^', 's', 'd', 'h', 'v', 'o', 'd']
    methods = ["DD-GCN_batch", "DEMD", "CR-MGC", "CEN", "HERO", "SIDR", "GCN_2017", "CEN_scaled"]
    method_labels = ["DD-GCN", "DEMD", "CR-MGC", "centering", "HERO", "SIDR", "GCN", "scaled-center"]

    for m in methods:
        step_m = []
        for dnum in num_destroyed:
            try:
                with open(f'./Logs/d{dnum}/{m}_d{dnum}.txt', 'r') as f:
                    data = f.read()

                data = data.split('\n')
                # print(data[7].replace(' ',''))
                step = float(data[7].replace(' ',''))
                # if m != "DEMD" and step < 499:
                #     step_m.append(float(data[7].replace(' ',''))+10)
                # else:
                #     step_m.append(float(data[7].replace(' ','')))
                step_m.append(float(data[7].replace(' ','')))
            except:
                step_m.append(499)

        print(step_m)
        step_methods.append(step_m)

    plt.figure()
    for i, s in enumerate(step_methods):
        plt.plot(num_destroyed, s, c=mcolors.TABLEAU_COLORS[colors[i]],marker=marklist[i],label=method_labels[i])

    plt.xlim(10,200)
    plt.ylim(0,540)
    plt.grid(linestyle='--')
    plt.xlabel('Number of destroyed UAVs', fontdict={'family':'serif', 'size':14})
    plt.ylabel('Average recovery steps $J_S$', fontdict={'family':'serif', 'size':14})
    plt.legend(loc='upper left')
    plt.plot(num_destroyed, step_methods[2], c=mcolors.TABLEAU_COLORS[colors[2]],marker=marklist[2])
    plt.plot(num_destroyed, step_methods[1], c=mcolors.TABLEAU_COLORS[colors[1]],marker=marklist[1])
    plt.plot(num_destroyed, step_methods[0], c=mcolors.TABLEAU_COLORS[colors[0]],marker=marklist[0])
    plt.show()

    ratio = []
    for i in range(7):
        ratio.append((step_methods[1][i] - step_methods[0][i])/step_methods[1][i])

    print(ratio, sum(ratio)/7)



def draw_spatial_coverage():
    colors = list(mcolors.TABLEAU_COLORS.keys())

    num_destroyed = [25, 50, 75, 100, 125, 150, 175]

    # coverage_methods = [[0.5790114182692307, 0.54329453125, 0.47705390624999994, 0.4067041666666667, 0.31975442708333335, 0.24608235677083334, 0.15416647135416667], [0.551693835136218, 0.5039954427083333, 0.45265026041666667, 0.36421516927083336, 0.27528483072916665, 0.20392389322916663, 0.13745338541666668], [0.5472358273237179, 0.5137605794270833, 0.42624472656250006, 0.3349951171875001, 0.23151835937499998, 0.1789016927083333, 0.10261953124999998]]
    # scaled
    # coverage_methods = [[0.5790114182692307, 0.54329453125, 0.47705390624999994, 0.4067041666666667, 0.31975442708333335, 0.24608235677083334, 0.15416647135416667], [0.551693835136218, 0.5039954427083333, 0.45265026041666667, 0.36421516927083336, 0.27528483072916665, 0.20392389322916663, 0.13745338541666668], [0.5873599008413461, 0.5707125, 0.5360072916666666, 0.4860467447916667, 0.39812076822916664, 0.2877386067708333, 0.17947519531249997]]
    coverage_methods = [[0.5968694911858974, 0.5925361328125, 0.5809194010416667, 0.5435434895833333, 0.45511796875, 0.3193609375, 0.1830646484375], [0.5790114182692307, 0.54329453125, 0.47705390624999994, 0.4067041666666667, 0.31975442708333335, 0.24608235677083334, 0.15416647135416667], [0.551693835136218, 0.5039954427083333, 0.45265026041666667, 0.36421516927083336, 0.27528483072916665, 0.20392389322916663, 0.13745338541666668], [0.5472358273237179, 0.5137605794270833, 0.42624472656250006, 0.34188515625, 0.23151835937499998, 0.17538046875, 0.10261953124999998], [0.5873599008413461, 0.5707125, 0.5360072916666666, 0.4860467447916667, 0.39812076822916664, 0.2877386067708333, 0.17947519531249997]]

    # coverage_methods = []
    marklist = ['o', '^', 's', 'd', 'v']
    methods = ["DD-GCN_batch", "DEMD", "CR-MGC", "CEN", "CEN_scaled"]
    method_labels = ["DD-GCN", "DEMD", "CR-MGC", "centering", "scaled-center"]

    # for m in methods:
    #     coverage_m = []
    #     for dnum in num_destroyed:
    #         with open(f'./Logs/d{dnum}/{m}_d{dnum}.txt', 'r') as f:
    #             data = f.read()

    #         data = data.split('\n')
            
    #         coverage_d = []

    #         for col in range(15, len(data), 5):
    #             pos = eval(data[col].replace('array', 'np.array'))

    #             plt.figure()
    #             for (x,y) in pos:
    #                 plt.gcf().gca().add_artist(plt.Circle((x,y), 120, color='#000000'))

    #             plt.xlim(0,1000)
    #             plt.ylim(0,1000)
    #             # plt.show()

    #             canvas = FigureCanvasAgg(plt.gcf())
    #             canvas.draw()

    #             w, h = canvas.get_width_height()

    #             buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)

    #             buf.shape = (w,h,4)
    #             buf = np.roll(buf, 3, axis=2)
    #             # print(np.sum(buf[:,:,1]==0)/(w*h))

    #             plt.close()

    #             coverage_d.append(np.sum(buf[:,:,1]==0)/(w*h))

    #         # print(data[col-2])
    #         # print(pos)
    #         coverage_m.append(np.mean(coverage_d))

    #     print(coverage_m)
    #     coverage_methods.append(coverage_m)
    
    plt.figure()
    for i, s in enumerate(coverage_methods):
        plt.plot(num_destroyed, s, c=mcolors.TABLEAU_COLORS[colors[i]],marker=marklist[i],label=method_labels[i])

    plt.xlim(10,200)
    plt.ylim(0.08,0.62)
    plt.grid(linestyle='--')
    plt.xlabel('Number of destroyed UAVs', fontdict={'family':'serif', 'size':14})
    plt.ylabel('Average spatial coverage ratio', fontdict={'family':'serif', 'size':14})
    plt.plot(num_destroyed, coverage_methods[0], c=mcolors.TABLEAU_COLORS[colors[0]],marker=marklist[0])
    plt.legend(loc='upper right')
    plt.show()

    ratio = []
    for i in range(7):
        ratio.append((coverage_methods[0][i] - coverage_methods[2][i])/coverage_methods[2][i])

    print(ratio, sum(ratio)/7)


def draw_degree_distribution():
    num_destroyed = [150]
    # num_destroyed = [25, 50, 75, 100, 125, 150, 175]

    methods = ["DD-GCN_L1", "DEMD", "CR-MGC", "CEN", "DF-scaled", "GCN_2017"]
    method_labels = ["DD-GCN", "DEMD", "CR-MGC", "centering", "scaled-center", "GCN"]

    for dnum in num_destroyed:
        drange = range(200-dnum)
        
        plt.figure()
        for i, m in enumerate(methods):
            with open(f'./Logs/d{dnum}/{m}_d{dnum}.txt', 'r') as f:
                data = f.read()

            data = data.split('\n')
            # print(data[7].replace(' ',''))
            degrees = eval(data[10])
            if m in ["CEN", "GCN_2017"] and dnum == 150:
                degrees = degrees[0:(200-dnum)*20]

            dcount = []

            for d in drange:
                # print(np.size(degrees))
                dcount.append(np.sum(np.array(degrees)<=d)/np.size(degrees))

            plt.plot(drange, dcount, label=method_labels[i])
        plt.grid(axis='y')
        plt.xlabel(f'Node degree $d$ with $N_D={dnum}$', fontdict={'family':'serif', 'size':14})
        plt.ylabel(f'Cumulative Degree Distribution $P_d$', fontdict={'family':'serif', 'size':14})
        plt.ylim(0, 1.03)
        plt.legend(loc='lower right')
        plt.show()
        # plt.savefig(f'./Figs/distribution/fig_d{dnum}.png')
        # print(step_m)
        # step_methods.append(step_m)


    # plt.figure()
    # for i, s in enumerate(step_methods):
    #     plt.plot(num_destroyed, s, c=mcolors.TABLEAU_COLORS[colors[i]],marker=marklist[i],label=method_labels[i])

    # plt.xlim(10,200)
    # plt.ylim(0,540)
    # plt.grid()
    # plt.xlabel('Number of destroyed UAVs', fontdict={'family':'serif', 'size':14})
    # plt.ylabel('Average recovery steps $J_S$', fontdict={'family':'serif', 'size':14})
    # plt.legend(loc='upper left')
    # plt.plot(num_destroyed, step_methods[2], c=mcolors.TABLEAU_COLORS[colors[2]],marker=marklist[2])
    # plt.plot(num_destroyed, step_methods[1], c=mcolors.TABLEAU_COLORS[colors[1]],marker=marklist[1])
    # plt.plot(num_destroyed, step_methods[0], c=mcolors.TABLEAU_COLORS[colors[0]],marker=marklist[0])
    # plt.show()



if __name__ == "__main__":
    # draw_method_figs()
    # draw_khop_figs()
    # draw_spatial_coverage()
    draw_degree_distribution()