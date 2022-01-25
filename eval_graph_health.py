import pickle

import matplotlib.pyplot as plt
import numpy as np
import statistics

import typing

accs_civb = []
dps_civb = []
cvib = np.load('health_other/cvib.npy', allow_pickle=True)
b = list(cvib.flatten()[0].values())
for item in b[1:]:
    accs_civb.append(item[0]['result']['nn_1_layer_normalized']['acc'][0])
    dps_civb.append(item[0]['result']['nn_1_layer_normalized']['dp'][0])

accs_adv_forgetting = []
dps_adv_forgetting = []
adv_forgetting = np.load('health_other/adv_forgetting.npy', allow_pickle=True)
b = list(adv_forgetting.flatten()[0].values())
for item in b[1:]:
    accs_adv_forgetting.append(item[0]['result']['nn_1_layer_normalized']['acc'][0])
    dps_adv_forgetting.append(item[0]['result']['nn_1_layer_normalized']['dp'][0])

accs_fcrl = []
dps_fcrl = []
fcrl = np.load('health_other/fcrl.npy', allow_pickle=True)
b = list(fcrl.flatten()[0].values())
for item in b[1:]:
    accs_fcrl.append(statistics.mean(item[0]['result']['nn_1_layer_normalized']['acc']))
    dps_fcrl.append(statistics.mean(item[0]['result']['nn_1_layer_normalized']['dp']))

accs_maxent = []
dps_maxent = []
maxent = np.load('health_other/maxent.npy', allow_pickle=True)
b = list(maxent.flatten()[0].values())
for item in b[1:]:
    accs_maxent.append(item[0]['result']['nn_1_layer_normalized']['acc'][0])
    dps_maxent.append(item[0]['result']['nn_1_layer_normalized']['dp'][0])

accs_mifr = []
dps_mifr = []
mifr = np.load('health_other/lag-fairness.npy', allow_pickle=True)
b = list(mifr.flatten()[0].values())
for item in b[1:]:
    accs_mifr.append(item[0]['result']['nn_1_layer_normalized']['acc'][0])
    dps_mifr.append(item[0]['result']['nn_1_layer_normalized']['dp'][0])

accs_att = [0.7637014, 0.7641484, 0.7650425, 0.76450604, 0.7637014, 0.76343316, 0.7628073, 0.7638802, 0.7637908,
            0.7657577, 0.76307553, 0.76450604, 0.7652213, 0.7653107, 0.7653107, 0.76244974, 0.764059, 0.76557887,
            0.76343316, 0.7620921, 0.7617345, 0.764059, 0.7654001, 0.7621815, 0.7653107, 0.7653107, 0.7649531,
            0.76477426, 0.76486367, 0.76548946, 0.763612, 0.7653107, 0.7657577, 0.76486367, 0.76557887, 0.76459545,
            0.7654001, 0.7656683, 0.7650425, 0.76128745, 0.7620921, 0.7651319, 0.76486367, 0.76459545, 0.7656683,
            0.7650425, 0.7658471, 0.76441664, 0.7651319, 0.76334375, 0.7638802, 0.7638802, 0.76316494, 0.7643272,
            0.76548946, 0.764059, 0.76548946, 0.7653107, 0.7620921, 0.7656683, 0.7652213, 0.7650425, 0.7604828,
            0.7639696, 0.76253915, 0.76110864, 0.76244974, 0.76325434, 0.7628073, 0.76307553, 0.76557887, 0.7641484,
            0.69503796]
dps_att = [0.5448804595759061, 0.5111937146236378, 0.535513643659711, 0.5415962659457633, 0.5593794880459576,
           0.5458097490918307, 0.5373933429078314, 0.5331692996536285, 0.5364640533919067, 0.5434759651938836,
           0.5359994086339444, 0.5411421812959365, 0.5570457041480105, 0.5570457041480105, 0.5570457041480105,
           0.5266431528258849, 0.5111937146236378, 0.5570457041480105, 0.49715933091154857, 0.4989492297237048,
           0.5331904198698995, 0.5369392582580046, 0.5575209090141083, 0.5430007603277858, 0.5570457041480105,
           0.555641209765988, 0.5570457041480105, 0.5514277266199206, 0.5589148432879952, 0.5584607586381685,
           0.5537615105178677, 0.5547013601419278, 0.5593900481540931, 0.5570457041480105, 0.5542367153839655,
           0.555641209765988, 0.5570457041480105, 0.5542367153839655, 0.555641209765988, 0.5434759651938836,
           0.5401917715637408, 0.5364429331756357, 0.5523675762439807, 0.5640681760581228, 0.5575209090141083,
           0.5542367153839655, 0.5622201571344091, 0.555641209765988, 0.555641209765988, 0.5336656247359973,
           0.5542261552758301, 0.5518923713778829, 0.5556306496578526, 0.5626636816761004, 0.5570457041480105,
           0.5542367153839655, 0.558450198530033, 0.5144779082537805, 0.5401917715637408, 0.5561164146320858,
           0.5570457041480105, 0.5570457041480105, 0.5476788882318155, 0.5668666047140323, 0.5420609107037256,
           0.5135063783053139, 0.5266220326096139, 0.5537720706260032, 0.538322632423756, 0.5537509504097322,
           0.5598546929120554, 0.5579749936639351, 0.12359550561797752]
mlp_acc = [0.851050514081359]
mlp_dp = [0.876737]

def domination(x1, x2):
    # determin if x1 dominate x2
    # breakpoint()
    # want greater acc in 0 dim and lower dp in 1 dim
    if x1[0] > x2[0] and x1[1] < x2[1]:
        return True
    if x1[0] >= x2[0] and x1[1] < x2[1]:
        return True
    if x1[0] > x2[0] and x1[1] <= x2[1]:
        return True
    return False


def get_pareto_front(arr, x_dominates_y: typing.Callable = domination):
    pareto_front = []
    for i in arr:
        for j in arr:
            # print(i,j, x_dominates_y(j, i))
            # if j dominate i, we don't want i
            if x_dominates_y(j, i):
                # print("i is dominated by j")
                break
        else:
            pareto_front.append(i)
    return pareto_front


with open('acc_dps_health.pkl', 'rb') as fp:
    acc_dp = pickle.load(fp)

for item in acc_dp.keys():
    a = [(i, j) for i, j in zip(acc_dp[item][0], acc_dp[item][1])]
    a.sort(key=lambda x: -x[0])
    acc_dp[item] = a

pareto_set = [acc_dp[item] for item in acc_dp.keys()]
pareto_set_all = []
for item in pareto_set:
    pareto_set_all += item

pareto_set_all = get_pareto_front(pareto_set_all)

best_accs = [i[0] for i in pareto_set_all]
dps_best_acc = [i[1] for i in pareto_set_all]

del best_accs[-1]
del dps_best_acc[-1]

plt.figure(dpi=1200)
plt.plot(accs_fcrl, dps_fcrl, 'ro', label='FCRL', markerfacecolor='none', ms=6, markeredgecolor='purple')
plt.plot(accs_civb, dps_civb, 'bs', label='CVIB', markerfacecolor='none', ms=6, markeredgecolor='blue')
plt.plot(accs_maxent, dps_maxent, 'g^', label='MaxEnt-ARL', markerfacecolor='none', ms=6, markeredgecolor='green')
plt.plot(accs_adv_forgetting, dps_adv_forgetting, 'k*', label='Adversarial Forgetting', markerfacecolor='none', ms=6,
         markeredgecolor='orange')
plt.plot(accs_att, dps_att, 'kP', label='Attention', markerfacecolor='none', ms=6,
         markeredgecolor='cyan')
plt.plot(accs_mifr, dps_mifr, 'kX', label='MIFR', markerfacecolor='none', ms=6,
         markeredgecolor='gray')
plt.plot(mlp_acc, mlp_dp, 'kH', label='Unfair MLP', markerfacecolor='pink', ms=10,
         markeredgecolor='pink')
plt.plot(best_accs, dps_best_acc, 'kD', label='Distraction(Ours)', markerfacecolor='none', ms=6,
         markeredgecolor='red')

plt.xlabel('Accuracy')
plt.ylabel('Statistical Parity Difference')
plt.grid(True)
plt.legend(loc='upper left', ncol=2, fontsize="xx-small")
plt.savefig('health_compare_2.pdf')
