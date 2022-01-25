import pickle

import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy.integrate import simps

import typing

from metrics import demographic_parity_difference


accs_civb = []
dps_civb = []
cvib = np.load('adult_other/cvib.npy', allow_pickle=True)
b = list(cvib.flatten()[0].values())
for item in b[1:]:
    accs_civb.append(item[0]['result']['nn_2_layer_normalized']['acc'][0])
    dps_civb.append(item[0]['result']['nn_2_layer_normalized']['dp'][0])

print("civb" + str(np.trapz(dps_civb, accs_civb)))

accs_adv_forgetting = []
dps_adv_forgetting = []
adv_forgetting = np.load('adult_other/adv_forgetting.npy', allow_pickle=True)
b = list(adv_forgetting.flatten()[0].values())
for item in b[1:]:
    accs_adv_forgetting.append(item[0]['result']['nn_2_layer_normalized']['acc'][0])
    dps_adv_forgetting.append(item[0]['result']['nn_2_layer_normalized']['dp'][0])

print("adv_forgetting" + str(np.trapz(dps_adv_forgetting, accs_adv_forgetting)))

accs_fcrl = []
dps_fcrl = []
fcrl = np.load('adult_other/fcrl.npy', allow_pickle=True)
b = list(fcrl.flatten()[0].values())
for item in b[1:]:
    accs_fcrl.append(statistics.mean(item[0]['result']['nn_2_layer_normalized']['acc']))
    dps_fcrl.append(statistics.mean(item[0]['result']['nn_2_layer_normalized']['dp']))


print("fcrl" + str(np.trapz(dps_fcrl, accs_fcrl)))


accs_maxent = []
dps_maxent = []
maxent = np.load('adult_other/maxent_arl.npy', allow_pickle=True)
b = list(maxent.flatten()[0].values())
for item in b[1:]:
    accs_maxent.append(item[0]['result']['nn_2_layer_normalized']['acc'][0])
    dps_maxent.append(item[0]['result']['nn_2_layer_normalized']['dp'][0])

print("maxent_arl" + str(np.trapz(dps_maxent, accs_maxent)))


accs_mifr = []
dps_mifr = []
mifr = np.load('adult_other/lag-fairness.npy', allow_pickle=True)
b = list(mifr.flatten()[0].values())
for item in b[1:]:
    accs_mifr.append(item[0]['result']['nn_2_layer_normalized']['acc'][0])
    dps_mifr.append(item[0]['result']['nn_2_layer_normalized']['dp'][0])

print("lag-fairness" + str(np.trapz(dps_mifr, accs_mifr)))

# accs_distraction = [0.751, 0.8388888888888889, 0.8394444444444444, 0.8391111111111111, 0.8393333333333334,
#                     0.8395555555555556, 0.8398888888888889, 0.84, 0.8405555555555555, 0.8402222222222222,
#                     0.8403333333333334, 0.8411111111111111, 0.8415555555555555, 0.8422222222222222, 0.8428888888888889]
# dps_distraction = [0, 0.053765363133475894, 0.05611593398639732, 0.0547837914216041, 0.054621413893404955,
#                    0.05513062803746907, 0.05580589996964343, 0.05680592695821611, 0.05931151481951448,
#                    0.05963258961600168, 0.059806007923934174, 0.0597802461045564, 0.06043711673717525,
#                    0.06059949426537441, 0.06227479318612143]


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


with open('adult_different_alpha.pkl', 'rb') as fp:
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

print("distraction" + str(np.trapz(dps_best_acc, best_accs)))

accs_att = [
    0.84568393, 0.84555113, 0.8439575, 0.850996, 0.8385126, 0.8434263, 0.80405045, 0.8385126, 0.8440239, 0.84541833,
    0.8436919, 0.84528553, 0.8460159, 0.7885126
]
dps_att = [0.20135528191517924, 0.1880038179249462, 0.20539141182725074, 0.19639999517774645, 0.1798246019619872,
           0.2044277233762009, 0.05734628299635497,
           0.1784557960729164, 0.22526766164894874, 0.19647922937660678, 0.18052266528359195, 0.18256417955063217, 0.1992857047002389, 0.02132015769571384]

print("attenntion" + str(np.trapz(dps_att, accs_att)))


mlp_acc = [0.8484888918540263]
mlp_dp = [0.18820572]
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
plt.savefig('adult_compare.pdf')

