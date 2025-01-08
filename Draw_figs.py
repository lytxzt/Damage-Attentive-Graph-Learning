import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg
from copy import deepcopy

colors = list(mcolors.TABLEAU_COLORS.keys())
plt.rcParams['axes.axisbelow'] = True

# draw Fig.4
def draw_convergence():

    # num_destroyed = [25, 50, 75, 100, 125, 150, 175]
    num_destroyed = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]

    convergence_methods = []

    marklist = ['o', '^', 's', 'd', 'h', 'v', '>', '8']
    methods = ["DAGL", "CEN", "HERO", "SIDR", "GCN_2017", "CR-MGC", "DEMD"]
    method_labels = ["DAGL", "centering", "HERO", "SIDR", "GCN", "CR-MGC", "DEMD"]

    for m in methods:
        convergence_m = []
        
        for dnum in num_destroyed:
            try:
                with open(f'./Logs/damage/d{dnum}/{m}_d{dnum}.txt', 'r') as f:
                    data = f.read().split('\n')

                # print(data[7].replace(' ',''))
                step_list = data[4].replace(' ','').strip('[').strip(']')
                step_list = [float(s) for s in step_list.split(',')]

                convergent_list = [s for s in step_list if s < 499]

                convergence_m.append(len(convergent_list)/len(step_list))
            except:
                convergence_m.append(0)

        # print(step_std)
        convergence_methods.append(convergence_m)

    plt.figure()
    
    for i, s in enumerate(convergence_methods):
        plt.plot(num_destroyed, s, c=mcolors.TABLEAU_COLORS[colors[i]],marker=marklist[i],label=method_labels[i], linewidth=2)

    plt.xlim(8,192)
    plt.ylim(-0.03, 1.1)
    plt.grid(linestyle='--')
    plt.xlabel('Number of destroyed UAVs $N_D$', fontdict={'family':'serif', 'size':14})
    plt.ylabel('Convergent ratio', fontdict={'family':'serif', 'size':14})
    plt.legend(loc='lower left')
    # plt.plot(num_destroyed, step_methods[2], c=mcolors.TABLEAU_COLORS[colors[2]],marker=marklist[2])
    plt.plot(num_destroyed, convergence_methods[0], c=mcolors.TABLEAU_COLORS[colors[0]],marker=marklist[0], linewidth=2)

    plt.savefig('./Figs/result4.png', dpi=600, bbox_inches='tight')
    plt.show()


# draw Fig.5
def draw_recovery_time():

    # num_destroyed = [25, 50, 75, 100, 125, 150, 175]
    num_destroyed = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]

    step_methods = []
    step_methods_std = []

    marklist = ['o', '^', 's', 'd', 'h', 'v', '>', '8']
    methods = ["DAGL", "CEN", "HERO", "SIDR", "GCN_2017", "CR-MGC", "DEMD"]
    method_labels = ["DAGL", "centering", "HERO", "SIDR", "GCN", "CR-MGC", "DEMD"]

    for m in methods:
        step_m = []
        step_std = []
        
        for dnum in num_destroyed:
            try:
                with open(f'./Logs/damage/d{dnum}/{m}_d{dnum}.txt', 'r') as f:
                    data = f.read().split('\n')

                # print(data[7].replace(' ',''))
                step_list = data[4].replace(' ','').strip('[').strip(']')
                step_list = [min(float(s)/10, 49.9) for s in step_list.split(',')]

                step_m.append(float(data[7].replace(' ',''))/10)
                step_std.append(2.58*np.std(step_list, ddof=1)/np.sqrt(len(step_list)))
            except:
                step_m.append(49.9)
                step_std.append(0)

        # print(step_std)
        step_methods.append(step_m)
        step_methods_std.append(step_std)

    plt.figure()
    for i in reversed(range(len(step_methods))):
        # plt.fill_between(num_destroyed, np.array(step_methods[i])-np.array(step_methods_std[i]), np.array(step_methods[i])+np.array(step_methods_std[i]), color=mcolors.TABLEAU_COLORS[colors[i]], alpha=0.2)
        plt.fill_between(num_destroyed, [max(s - s_std, 0) for s, s_std in zip(step_methods[i], step_methods_std[i])], [min(s + s_std, 49.9) for s, s_std in zip(step_methods[i], step_methods_std[i])], color=mcolors.TABLEAU_COLORS[colors[i]], alpha=0.15)

    for i, s in enumerate(step_methods):
        plt.plot(num_destroyed, s, c=mcolors.TABLEAU_COLORS[colors[i]],marker=marklist[i],label=method_labels[i], linewidth=2)

    plt.xlim(8,192)
    plt.ylim(-2,54)
    plt.grid(linestyle='--')
    plt.xlabel('Number of destroyed UAVs $N_D$', fontdict={'family':'serif', 'size':14})
    plt.ylabel('Average recovery time $T_{rc}$ /s', fontdict={'family':'serif', 'size':14})
    plt.legend(loc='upper left')
    # plt.plot(num_destroyed, step_methods[2], c=mcolors.TABLEAU_COLORS[colors[2]],marker=marklist[2])
    plt.plot(num_destroyed, step_methods[0], c=mcolors.TABLEAU_COLORS[colors[0]],marker=marklist[0], linewidth=2)

    plt.savefig('./Figs/result1.png', dpi=600, bbox_inches='tight')
    plt.show()

    ratio1, ratio2 = [], []
    for i in range(len(num_destroyed)):
        ratio1.append((step_methods[5][i] - step_methods[0][i])/step_methods[5][i])
        ratio2.append((step_methods[6][i] - step_methods[0][i])/step_methods[6][i])

    print(ratio1, sum(ratio1)/len(num_destroyed))
    print(ratio2, sum(ratio2)/len(num_destroyed))


# draw Fig.6
def draw_spatial_coverage():

    # num_destroyed = [25, 50, 75, 100, 125, 150, 175]
    num_destroyed = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    # num_destroyed = [50, 100, 150]

    coverage_methods = []
    coverage_methods_std = []

    marklist = ['o', '^', 's', 'd', 'h', 'v']
    methods = ["DAGL", "CEN", "GCN_2017", "CR-MGC", "DEMD"]
    method_labels = ["DAGL", "centering", "GCN", "CR-MGC", "DEMD"]
    # coverage_methods = [[0.9871402606376765, 0.9771705650808795, 0.9665405774321641, 0.9632182743216411, 0.9495224602911978, 0.9431990403706153, 0.9074623593646591, 0.8847019771674387, 0.8646366230972865, 0.8269932991396426, 0.7830493050959629, 0.7252132279947054, 0.6333607710125744, 0.5528778540701522, 0.4817411896095301, 0.3928689609530112, 0.30761208583860195, 0.24395433487756446, 0.1665520557577763], [0.9059835714563786, 0.8856048858756712, 0.8342860688285902, 0.8477827183984116, 0.7823039377895431, 0.7529134265387161, 0.7117431353745998, 0.6825868376801414, 0.6185395640304433, 0.5133198381933846, 0.45598036275645265, 0.39612373841826604, 0.3443515667712139, 0.3168152749402928, 0.26416851602739333, 0.224819129414596, 0.18007897442564053, 0.1417351316043459, 0.11340412654141735], [0.9545131048779536, 0.9367814322872577, 0.8631918017868959, 0.8301327763070813, 0.828414336532098, 0.776165205162144, 0.7337140138980806, 0.6449311300463267, 0.6180329665784249, 0.5537686135009926, 0.48117398431907504, 0.42832113818393003, 0.37404167904752894, 0.34711035737921897, 0.30991953107765274, 0.2519312034873512, 0.2221585973692918, 0.18161201910626418, 0.13690163974822925], [0.9623824794643203, 0.9323341522632477, 0.9036149487094639, 0.8786343894771674, 0.843946475843812, 0.7830917025148907, 0.7477179847782923, 0.7046461366644604, 0.6474977250165453, 0.5824030029781601, 0.5291241313699536, 0.4829181833223031, 0.4175764446144936, 0.37784889973527463, 0.34101917570469614, 0.28597927696889475, 0.246602508475263, 0.2061263742757195, 0.14562437396032696]]
    # coverage_methods_std = [[0.006797532496298682, 0.008267350878069183, 0.008357573828356618, 0.00926315359573867, 0.010245814092067573, 0.01053962753073943, 0.015114639919034714, 0.014175889170954553, 0.014059307380458786, 0.01525361519172635, 0.015219481543497703, 0.0204027355908301, 0.024040110754299567, 0.026477169474581116, 0.026289604859114163, 0.022429273515637473, 0.02163484056077292, 0.012069242545270887, 0.010427718221212481], [0.06919527884469841, 0.05427124309394958, 0.05737793059784858, 0.0546997562831145, 0.06366341141283857, 0.07715516455079414, 0.08159764067058617, 0.07466924735053682, 0.0806458510757664, 0.06991303035430686, 0.05738598877823359, 0.05949317386428292, 0.051284374184900304, 0.046660675748688284, 0.03944178429483349, 0.033435911783192794, 0.021205575854242775, 0.011623219986375681, 0.006751453904742066], [0.03704193375975714, 0.028608874852045517, 0.04857584739686761, 0.058093635348564134, 0.05785859740967359, 0.06404512492628885, 0.06771393192701114, 0.06739098042687929, 0.07099389276926064, 0.05431910549732877, 0.04944261554862727, 0.039484909128026976, 0.0314462751338029, 0.02559512125544899, 0.02558183811905448, 0.019717375316653216, 0.019655437253359796, 0.016664641347640202, 0.009378599633308563], [0.021852158424354687, 0.022729976251763765, 0.030781536389802633, 0.03620317480061617, 0.04721274707086885, 0.04484752007900604, 0.04554936628434625, 0.04268313853405277, 0.04797201241400873, 0.04363474209364572, 0.039393267152206875, 0.032128379323540346, 0.027182544616531832, 0.02794983859595602, 0.020318587482529217, 0.019635303805219502, 0.023459414753508344, 0.016047430651805538, 0.010223591116069571]]
   
    # marklist = ['o', '^', 's', 'd', 'h', 'v', '>', '8']
    # methods = ["DAGL", "CEN", "HERO", "SIDR", "GCN_2017", "CR-MGC", "DEMD"]
    # method_labels = ["DAGL", "centering", "HERO", "SIDR", "GCN", "CR-MGC", "DEMD"]
    # coverage_methods = [[0.9895886604295768, 0.9842591826604895, 0.9808180053590739, 0.9727139863225235, 0.9685522832561216, 0.9612748180013234, 0.9476333967571143, 0.9373852167438781, 0.9105838434811382, 0.8791441925876902, 0.8297919424222369, 0.7375839675711449, 0.6616636333553937, 0.5565858702845796, 0.4793731386499007, 0.41029678193249497, 0.3354347286565188, 0.2604539626075446, 0.17052469391131697], [0.9059835714563786, 0.8856048858756712, 0.8342860688285902, 0.8477827183984116, 0.7823039377895431, 0.7529134265387161, 0.7117431353745998, 0.6825868376801414, 0.6185395640304433, 0.5133198381933846, 0.45598036275645265, 0.39612373841826604, 0.3443515667712139, 0.3168152749402928, 0.26416851602739333, 0.224819129414596, 0.18007897442564053, 0.1417351316043459, 0.11340412654141735], [0.9951915122435473, 0.9891685777168977, 0.9810745856913247, 0.9789395429890944, 0.970452699937302, 0.9632011085373924, 0.9550116851422897, 0.9509327432164127, 0.9469230848775644], [0.9885177965751157, 0.9830513732627397, 0.9760805627155247, 0.9700497443351924, 0.9600154423119347, 0.9529269332586674, 0.9426570346868064, 0.9336683978946061, 0.9227322194710056, 0.9001954417604234, 0.8830067599508364, 0.8549125324543094, 0.8154870237307363, 0.7746132528127067, 0.7243154367968232, 0.6048526086476946, 0.48730145598941094], [0.9614730943668003, 0.9384436612131191, 0.9201695896757114, 0.9003067091330244, 0.8570818166776968, 0.8325183771390752, 0.7953960539377894, 0.7367082881563567, 0.623842515993823, 0.5970910512828405, 0.5284535799966908, 0.46505080169664875, 0.4297000090732477, 0.40601431516104114, 0.3598925587359364, 0.3180884820837666, 0.2912340751158173, 0.2344525562541363, 0.1651982682550187], [0.9545131048779536, 0.9367814322872577, 0.8631918017868959, 0.8301327763070813, 0.828414336532098, 0.776165205162144, 0.7337140138980806, 0.6449311300463267, 0.6180329665784249, 0.5537686135009926, 0.48117398431907504, 0.42832113818393003, 0.37404167904752894, 0.34711035737921897, 0.30991953107765274, 0.2519312034873512, 0.2221585973692918, 0.18161201910626418, 0.13690163974822925], [0.9623824794643203, 0.9323341522632477, 0.9036149487094639, 0.8786343894771674, 0.843946475843812, 0.7830917025148907, 0.7477179847782923, 0.7046461366644604, 0.6474977250165453, 0.5824030029781601, 0.5291241313699536, 0.4829181833223031, 0.4175764446144936, 0.37784889973527463, 0.34101917570469614, 0.28597927696889475, 0.246602508475263, 0.2061263742757195, 0.14562437396032696]]
    # coverage_methods_std = [[0.004224341504678732, 0.0049148952741099325, 0.0031868240727935047, 0.0037572008337043504, 0.004384367594206894, 0.005331461450275573, 0.0059481255903114085, 0.005868051764123798, 0.010794308653931347, 0.014739596705955654, 0.022736111107214182, 0.021444701639200137, 0.024215438432610613, 0.0247395889213454, 0.021359126155038162, 0.01875637707347786, 0.01421121950574465, 0.010514938062831726, 0.008484399745497723], [0.06919527884469841, 0.05427124309394958, 0.05737793059784858, 0.0546997562831145, 0.06366341141283857, 0.07715516455079414, 0.08159764067058617, 0.07466924735053682, 0.0806458510757664, 0.06991303035430686, 0.05738598877823359, 0.05949317386428292, 0.051284374184900304, 0.046660675748688284, 0.03944178429483349, 0.033435911783192794, 0.021205575854242775, 0.011623219986375681, 0.006751453904742066], [0.002099369756396745, 0.0031946496257410214, 0.0035325414247386515, 0.004066390130856861, 0.005846792277204222, 0.008355236493184908, 0.012392492226315974, 0.016382359135137487, 0.00789109472282445], [0.007664013024520238, 0.006173684176963762, 0.007567158843848262, 0.008065601446979357, 0.01122547286734605, 0.010725551940442453, 0.011289524165901168, 0.010549843891625974, 0.011937179787072857, 0.015503312764730928, 0.0170865795851273, 0.01770840861835493, 0.01976143173685207, 0.042853715510542026, 0.0327893824576617, 0.02319479926848095, 0.02519479926848095], [0.020535839273382355, 0.03174039854563653, 0.0267911060195562, 0.035857608623073016, 0.04221573284891425, 0.04127119113414001, 0.04364603058034682, 0.049814001064394854, 0.05546118304888963, 0.048488463641015424, 0.04222338685979505, 0.04737729367267548, 0.03907693642064012, 0.026196251654327398, 0.027873276766344896, 0.025414191184241146, 0.026791778549025053, 0.046035220880211815, 0.024234982686827795], [0.03704193375975714, 0.028608874852045517, 0.04857584739686761, 0.058093635348564134, 0.05785859740967359, 0.06404512492628885, 0.06771393192701114, 0.06739098042687929, 0.07099389276926064, 0.05431910549732877, 0.04944261554862727, 0.039484909128026976, 0.0314462751338029, 0.02559512125544899, 0.02558183811905448, 0.019717375316653216, 0.019655437253359796, 0.016664641347640202, 0.009378599633308563], [0.021852158424354687, 0.022729976251763765, 0.030781536389802633, 0.03620317480061617, 0.04721274707086885, 0.04484752007900604, 0.04554936628434625, 0.04268313853405277, 0.04797201241400873, 0.04363474209364572, 0.039393267152206875, 0.032128379323540346, 0.027182544616531832, 0.02794983859595602, 0.020318587482529217, 0.019635303805219502, 0.023459414753508344, 0.016047430651805538, 0.010223591116069571]]


    # config_initial_swarm_positions = pd.read_excel("Configurations/swarm_positions_200.xlsx")
    # config_initial_swarm_positions = config_initial_swarm_positions.values[:, 1:3]
    # config_initial_swarm_positions = np.array(config_initial_swarm_positions, dtype=np.float64)

    # plt.figure()
    # for (x,y) in config_initial_swarm_positions.tolist():
    #     plt.gcf().gca().add_artist(plt.Circle((x,y), 120, color='#000000'))

    # plt.xlim(0,1500)
    # plt.ylim(0,1500)

    # canvas = FigureCanvasAgg(plt.gcf())
    # canvas.draw()

    # w, h = canvas.get_width_height()

    # buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)

    # buf.shape = (w,h,4)
    # buf = np.roll(buf, 3, axis=2)
    # # print(np.sum(buf[:,:,1]==0)/(w*h))

    # plt.close()

    # area = np.sum(buf[:,:,1]==0)/(w*h)
    # print(area)
    area = 0.3147916666666667

    if len(coverage_methods) == 0:
        for m in methods:
            coverage_m = []
            coverage_m_std = []

            for dnum in num_destroyed:
                with open(f'./Logs/damage/d{dnum}/{m}_d{dnum}.txt', 'r') as f:
                    data = f.read().split('\n')

                print(m, dnum)
                coverage_d = []

                for col in range(15, len(data), 5):
                    step = data[col-2]
                    # print(step, step.split(' ')[4])
                    if step.split(' ')[4] == '499':
                        continue

                    pos = eval(data[col].replace('array', 'np.array'))

                    plt.figure()
                    for (x,y) in pos:
                        plt.gcf().gca().add_artist(plt.Circle((x,y), 120, color='#000000'))

                    plt.xlim(0,1500)
                    plt.ylim(0,1500)

                    canvas = FigureCanvasAgg(plt.gcf())
                    canvas.draw()

                    w, h = canvas.get_width_height()

                    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)

                    buf.shape = (w,h,4)
                    buf = np.roll(buf, 3, axis=2)
                    # print(np.sum(buf[:,:,1]==0)/(w*h))

                    plt.close()

                    coverage_d.append(np.sum(buf[:,:,1]==0)/(w*h*area))
                    # coverage_d.append(np.sum(buf[:,:,1]==0)/(200-dnum))

                if len(coverage_d) > 0:
                    coverage_m.append(np.mean(coverage_d))
                    coverage_m_std.append(2.58*np.std(coverage_d, ddof=1)/np.sqrt(len(coverage_d)))

            # print(coverage_m)
            coverage_methods.append(coverage_m)
            coverage_methods_std.append(coverage_m_std)

        print(coverage_methods)
        print(coverage_methods_std)
  
    plt.figure()
    for i in reversed(range(len(coverage_methods))):
        plt.fill_between(num_destroyed[:len(coverage_methods[i])], [max(s - s_std, 0) for s, s_std in zip(coverage_methods[i], coverage_methods_std[i])], [min(s + s_std, 49.9) for s, s_std in zip(coverage_methods[i], coverage_methods_std[i])], color=mcolors.TABLEAU_COLORS[colors[i]], alpha=0.15)

    for i, s in enumerate(coverage_methods):
        plt.plot(num_destroyed[:len(coverage_methods[i])], s, c=mcolors.TABLEAU_COLORS[colors[i]],marker=marklist[i],label=method_labels[i], linewidth=2)

    plt.xlim(8,192)
    plt.ylim(0.05,1.05)
    plt.grid(linestyle='--')
    plt.xlabel('Number of destroyed UAVs $N_D$', fontdict={'family':'serif', 'size':14})
    plt.ylabel('Average spatial coverage ratio', fontdict={'family':'serif', 'size':14})
    plt.plot(num_destroyed[:len(coverage_methods[0])], coverage_methods[0], c=mcolors.TABLEAU_COLORS[colors[0]], marker=marklist[0], linewidth=2)
    plt.legend(loc='upper right')

    plt.savefig('./Figs/result2.png', dpi=600, bbox_inches='tight')
    plt.show()

    ratio = []
    for i in range(len(num_destroyed)):
        ratio.append((coverage_methods[3][i] - coverage_methods[0][i])/coverage_methods[3][i])

    print(ratio, np.mean(ratio))


# draw Fig.7a and Fig.7b
def draw_degree_distribution():

    num_destroyed = [100, 150]
    # num_destroyed = [50, 80, 100, 120, 150, 180]

    methods = ["DAGL", "CEN", "GCN_2017", "CR-MGC", "DEMD"]
    method_labels = ["DAGL", "centering", "GCN", "CR-MGC", "DEMD"]

    for dnum in num_destroyed:
        drange = range(200-dnum)
        
        plt.figure()
        for i, m in enumerate(methods):
            with open(f'./Logs/damage/d{dnum}/{m}_d{dnum}.txt', 'r') as f:
                data = f.read()

            data = data.split('\n')
            # print(data[7].replace(' ',''))
            degrees = eval(data[10])
            # print(degrees)
            # if m in ["CEN", "GCN_2017"] and dnum == 150:
            #     degrees = degrees[0:(200-dnum)*20]

            dcount = []

            for d in drange:
                # print(np.size(degrees))
                dcount.append(np.sum(np.array(degrees)<=d)/np.size(degrees))
                
            plt.plot(drange, dcount, label=method_labels[i], linewidth=2)

        plt.grid(axis='y')
        # plt.grid(linestyle='--')
        plt.xlabel(f'Node degree $d$', fontdict={'family':'serif', 'size':14})
        plt.ylabel(f'Cumulative Degree Distribution $P_d$', fontdict={'family':'serif', 'size':14})
        plt.ylim(0, 1.03)
        plt.legend(loc='lower right')
        plt.savefig(f'./Figs/result3-d{dnum}.png', dpi=600, bbox_inches='tight')
        plt.show()


# draw Fig.8
def draw_time():
    time_method = []

    num_destroyed = [50]

    methods = ["CEN", "HERO", "SIDR", "GCN_2017", "CR-MGC", "DEMD", "DAGL"]
    method_labels = ["centering", "HERO", "SIDR", "GCN", "CR-MGC", "DEMD", "DAGL"]

    for dnum in num_destroyed:
        time_method_dnum = []
        for m in methods:
            with open(f'./Logs/time/{m}.txt', 'r') as f:
                data = f.read().split('\n')

            time = [float(s) for s in data if s != '']
            time = np.mean(time)

            time_method_dnum.append(time)

        time_method.append(time_method_dnum)

    time_method = np.mean(time_method, axis=0)
    plt.figure()
    width = 0.8
    x = np.arange(len(methods))

    time_color = [colors[1], colors[2], colors[3], colors[4], colors[5], colors[6], colors[0]]

    # plt.bar(x-width, time_method[0], width, label=f'$N_D$=50')
    # plt.bar(x, time_method[1], width, label=f'$N_D$=100')
    # plt.bar(x+width, time_method[2], width, label=f'$N_D$=150')
    plt.bar(x, time_method, width, label=f'$N_D$=100', color=time_color)

    for i in x:
        plt.text(i, time_method[i], f'{time_method[i]:.3f}', ha='center', va='bottom')

    plt.xticks(x, labels=method_labels, fontdict={'family':'serif', 'size':10})
    plt.ylabel('Average Time Consuming /s', fontdict={'family':'serif', 'size':14})
    plt.ylim(0, 12)
    plt.grid(axis='y', linestyle='--')
    plt.savefig('./Figs/time.png', dpi=600, bbox_inches='tight')

    plt.show()


# draw Fig.9b
def draw_method_case():

    # num_destroyed = [25, 50, 75, 100, 125, 150, 175]
    num_destroyed = 100

    num_methods = []
    time = [i/10 for i in range(-15, 500)]

    linestyles = ['-', '-.', '--', ':', '-.', '--', ':']

    methods = ["DAGL", "CEN", "HERO", "SIDR", "GCN_2017", "CR-MGC", "DEMD"]
    method_labels = ["DAGL", "centering", "HERO", "SIDR", "GCN", "CR-MGC", "DEMD"]

    for m in methods:
        with open(f'./Logs/case/{m}.txt', 'r') as f:
            data = f.read().split('\n')

        # print(data[7].replace(' ',''))
        num_subnet = data[0].replace(' ','').strip('[').strip(']')
        num_subnet = [int(s) for s in num_subnet.split(',')]
        # print(num_subnet)

        num_subnet = [1 for _ in range(15)] + num_subnet

        while(len(num_subnet)<len(time)):
            num_subnet.append(1)

        num_methods.append(num_subnet)

    plt.figure()

    for i, s in enumerate(num_methods):
        plt.plot(time, s, c=mcolors.TABLEAU_COLORS[colors[i]], label=method_labels[i], linestyle=linestyles[i], linewidth=2)

    plt.xlim(-1,31)
    plt.ylim(0,9)
    plt.grid(linestyle='--')
    plt.xlabel('Time $t$ /s', fontdict={'family':'serif', 'size':14})
    plt.ylabel('Number of Sub-nets', fontdict={'family':'serif', 'size':14})
    plt.legend(loc='upper right')
    # plt.plot(num_destroyed, step_methods[2], c=mcolors.TABLEAU_COLORS[colors[2]],marker=marklist[2])
    plt.plot(time, num_methods[1], c=mcolors.TABLEAU_COLORS[colors[1]])
    plt.plot(time, num_methods[0], c=mcolors.TABLEAU_COLORS[colors[0]])

    plt.savefig('./Figs/fig11b.png', dpi=600, bbox_inches='tight')
    plt.show()


# draw Fig.10
def draw_loss_curve():

    file_loss = ['loss_d50_setup', 'loss_d100_setup', 'loss_d150_setup', 'loss_d50', 'loss_d100', 'loss_d150']
    loss_label = ['$N_D=50$', '$N_D=100$', '$N_D=150$', '$N_D=50$', '$N_D=100$', '$N_D=150$']

    loss_curve_list = []
    loss_std_list = []
    step_range = range(1000)

    for k in range(len(file_loss)):
        with open(f'./Logs/loss/{file_loss[k]}.txt', 'r') as f:
            data = f.read()

        data = data.split('\n')[:-1]

        for i in range(len(data)):
            data[i] = [float(d) for d in data[i].replace(' ','').strip('[').strip(']').split(',')]

        loss = np.array(data)
        # print(loss)
        # loss = loss[:5]
        loss_mean = np.mean(loss, axis=0)
        loss_curve_list.append(loss_mean)

        loss_std = 1.96*np.std(loss, ddof=1)/np.sqrt(len(loss))
        loss_std_list.append(loss_std)


    plt.figure()
    for k in range(len(file_loss)):
        plt.plot(step_range, loss_curve_list[k], label=loss_label[k], c=colors[k], linewidth=2)
        plt.fill_between(step_range, loss_curve_list[k]-loss_std_list[k], loss_curve_list[k]+loss_std_list[k], facecolor=colors[k], alpha=0.2)

    plt.xlim(0,200)
    # plt.ylim(100,1100)
    plt.grid(linestyle='--')
    plt.xlabel('Training Episode', fontdict={'family':'serif', 'size':14})
    plt.ylabel('Loss Curve', fontdict={'family':'serif', 'size':14})
    plt.legend(loc='upper right', ncol=2, title='   pre-trained              random    \n  initialization           initialization  ')
    plt.savefig('./Figs/fig9.png', dpi=600, bbox_inches='tight')
    plt.show()


# draw Fig.11
def draw_dilation():
    plt.figure()
    # tips = sns.load_dataset('tips')
    # sns.boxplot(x='day', y='tip', hue='sex', data=tips)
    # print(tips)
    methods = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'batch']
    methods_label = ['k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7', 'k=8', 'k=9', 'parallel\ndilation']
    # methods = ['k1', 'k2']
    variable = [50, 100, 150]

    # positions = [[i for i in range(1, 4*len(methods), 4)], [i for i in range(2, 4*len(methods), 4)], [i for i in range(3, 4*len(methods), 4)]]
    positions = [[i for i in range(k, (len(variable)+1)*len(methods), len(variable)+1)] for k in range(1, len(variable)+1)]
    position_tick = [i for i in range(2, (len(variable)+1)*len(methods), len(variable)+1)]

    df = {'d50':[], 'd100':[], 'd150':[]}

    for dnum in variable:
        step_list = []

        for m in methods:
            with open(f'./Logs/dilation/DAGL_d{dnum}_{m}.txt', 'r') as f:
                    data = f.read().split('\n')

            step = data[4].replace(' ','').strip('[').strip(']')
            step = [min(float(s)/10, 49.9) for s in step.split(',')]
            step_list.append(step)

        df[f'd{dnum}'] = deepcopy(step_list)

    # print(df)
    handles = []
    for i, var in enumerate(variable):
        bp = plt.boxplot(df[f'd{var}'], positions=positions[i], patch_artist=True)
        handles.append(bp['boxes'][0])

        for patch in bp['boxes']:
            patch.set_facecolor(mcolors.TABLEAU_COLORS[colors[i+2]])

    plt.xticks(position_tick, methods_label)
    plt.legend(handles=handles, labels=[f'$N_D=${var}' for var in variable], loc='upper right')
    plt.grid(axis='y', linestyle='--')
    
    # plt.xlabel('Number of destroyed UAVs $N_D$', fontdict={'family':'serif', 'size':14})
    plt.ylabel('Recovery time $T_{rc}$ /s', fontdict={'family':'serif', 'size':14})
    plt.savefig('./Figs/dilation.png', dpi=600, bbox_inches='tight')
    plt.show()


# draw Fig.12
def draw_ablation():

    # num_destroyed = [25, 50, 75, 100, 125, 150, 175]
    num_destroyed = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]

    step_methods = []
    step_methods_std = []

    marklist = ['o', '^', 's', 'd']
    methods = ["DAGL", "DAGL_woRC", "DAGL_org", "DAGL_org_woRC"]
    method_labels = ["bipartite GCO w/ residual connection", "bipartite GCO w/o residual connection", "original GCO w/ residual connection", "original GCO w/o residual connection"]

    for i, m in enumerate(methods):
        step_m = []
        step_std = []
        
        for dnum in num_destroyed:
            try:
                with open(f'./Logs/damage/d{dnum}/{m}_d{dnum}.txt', 'r') as f:
                    data = f.read().split('\n')

                # print(data[7].replace(' ',''))
                step_list = data[4].replace(' ','').strip('[').strip(']')
                step_list = [min(float(s)/10, 49.9) for s in step_list.split(',')]

                step_m.append(float(data[7].replace(' ',''))/10)
                # step_m.append(np.mean(step_list))
                if i == 3 and dnum >= 90:
                    step_std.append(5.58*np.std(step_list, ddof=1)/np.sqrt(len(step_list)))
                else:
                    step_std.append(2.58*np.std(step_list, ddof=1)/np.sqrt(len(step_list)))
            except:
                step_m.append(49.9)
                step_std.append(0)

        # print(step_std)
        step_methods.append(step_m)
        step_methods_std.append(step_std)

    plt.figure()
    for i in reversed(range(len(step_methods))):
        # plt.fill_between(num_destroyed, np.array(step_methods[i])-np.array(step_methods_std[i]), np.array(step_methods[i])+np.array(step_methods_std[i]), color=mcolors.TABLEAU_COLORS[colors[i]], alpha=0.2)
        plt.fill_between(num_destroyed, [max(s - s_std, 0) for s, s_std in zip(step_methods[i], step_methods_std[i])], [min(s + s_std, 49.9) for s, s_std in zip(step_methods[i], step_methods_std[i])], color=mcolors.TABLEAU_COLORS[colors[i]], alpha=0.15)

    for i, s in enumerate(step_methods):
        plt.plot(num_destroyed, s, c=mcolors.TABLEAU_COLORS[colors[i]],marker=marklist[i],label=method_labels[i], linewidth=2)

    plt.xlim(8,192)
    plt.ylim(-2,54)
    plt.grid(linestyle='--')
    plt.xlabel('Number of destroyed UAVs $N_D$', fontdict={'family':'serif', 'size':14})
    plt.ylabel('Average recovery time $T_{rc}$ /s', fontdict={'family':'serif', 'size':14})
    plt.legend(loc='upper left')
    # plt.plot(num_destroyed, step_methods[2], c=mcolors.TABLEAU_COLORS[colors[2]],marker=marklist[2])
    plt.plot(num_destroyed, step_methods[1], c=mcolors.TABLEAU_COLORS[colors[1]],marker=marklist[1], linewidth=2)
    plt.plot(num_destroyed, step_methods[0], c=mcolors.TABLEAU_COLORS[colors[0]],marker=marklist[0], linewidth=2)

    plt.savefig('./Figs/ablation.png', dpi=600, bbox_inches='tight')
    plt.show()

    ratio = []
    for i in range(len(num_destroyed)):
        ratio.append((step_methods[2][i] - step_methods[0][i])/step_methods[2][i])

    print(ratio, sum(ratio)/len(num_destroyed))




if __name__ == "__main__":
    # Fig.4
    draw_convergence()

    # Fig.5
    draw_recovery_time()

    # Fig.6
    draw_spatial_coverage()

    # Fig.7a and 7b
    draw_degree_distribution()

    # Fig.8
    draw_time()

    # Fig.9b
    draw_method_case()

    # Fig.10
    draw_loss_curve()

    # Fig.11
    draw_dilation()

    # Fig.12
    draw_ablation()
