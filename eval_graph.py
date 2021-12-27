import matplotlib.pyplot as plt
import numpy as np
import statistics

accs_civb = []
dps_civb = []
cvib = np.load('adult_other/cvib.npy', allow_pickle=True)
b = list(cvib.flatten()[0].values())
for item in b[1:]:
    accs_civb.append(item[0]['result']['nn_1_layer_normalized']['acc'][0])
    dps_civb.append(item[0]['result']['nn_1_layer_normalized']['dp'][0])

accs_adv_forgetting = []
dps_adv_forgetting = []
adv_forgetting = np.load('adult_other/adv_forgetting.npy', allow_pickle=True)
b = list(adv_forgetting.flatten()[0].values())
for item in b[1:]:
    accs_adv_forgetting.append(item[0]['result']['nn_1_layer_normalized']['acc'][0])
    dps_adv_forgetting.append(item[0]['result']['nn_1_layer_normalized']['dp'][0])

accs_fcrl = []
dps_fcrl = []
fcrl = np.load('adult_other/fcrl.npy', allow_pickle=True)
b = list(fcrl.flatten()[0].values())
for item in b[1:]:
    accs_fcrl.append(statistics.mean(item[0]['result']['nn_1_layer_normalized']['acc']))
    dps_fcrl.append(statistics.mean(item[0]['result']['nn_1_layer_normalized']['dp']))

accs_maxent = []
dps_maxent = []
maxent = np.load('adult_other/maxent_arl.npy', allow_pickle=True)
b = list(maxent.flatten()[0].values())
for item in b[1:]:
    accs_maxent.append(item[0]['result']['nn_1_layer_normalized']['acc'][0])
    dps_maxent.append(item[0]['result']['nn_1_layer_normalized']['dp'][0])

accs_mifr = []
dps_mifr = []
mifr = np.load('adult_other/lag-fairness.npy', allow_pickle=True)
b = list(mifr.flatten()[0].values())
for item in b[1:]:
    accs_mifr.append(item[0]['result']['nn_1_layer_normalized']['acc'][0])
    dps_mifr.append(item[0]['result']['nn_1_layer_normalized']['dp'][0])
accs_distraction = [0.751, 0.8388888888888889, 0.8394444444444444, 0.8391111111111111, 0.8393333333333334,
                    0.8395555555555556, 0.8398888888888889, 0.84, 0.8405555555555555, 0.8402222222222222,
                    0.8403333333333334, 0.8411111111111111, 0.8415555555555555, 0.8422222222222222, 0.8428888888888889,
                    0.843, 0.8443333333333334, 0.8454444444444444, 0.8453333333333334]
dps_distraction = [0, 0.053765363133475894, 0.05611593398639732, 0.0547837914216041, 0.054621413893404955,
                   0.05513062803746907, 0.05580589996964343, 0.05680592695821611, 0.05931151481951448,
                   0.05963258961600168, 0.059806007923934174, 0.0597802461045564, 0.06043711673717525,
                   0.06059949426537441, 0.06227479318612143, 0.07058872337136789, 0.07174008710840639,
                   0.09299593007863041, 0.09139791661601662]

accs_att = [
    0.84568393, 0.84555113, 0.8439575, 0.850996, 0.8385126, 0.8434263, 0.80405045, 0.8385126, 0.8440239, 0.84541833,
    0.8436919, 0.84528553, 0.8460159, 0.7885126
]
dps_att = [0.20135528191517924, 0.1880038179249462, 0.20539141182725074, 0.19639999517774645, 0.1798246019619872,
           0.2044277233762009, 0.05734628299635497,
           0.1784557960729164, 0.22526766164894874, 0.19647922937660678, 0.18052266528359195, 0.18256417955063217, 0.1992857047002389, 0.02132015769571384]

mlp_acc = [0.846]
mlp_dp = [0.165]
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
plt.plot(accs_distraction, dps_distraction, 'kD', label='Distraction(Ours)', markerfacecolor='none', ms=6,
         markeredgecolor='red')

plt.xlabel('Accuracy')
plt.ylabel('Statistical Parity Difference')
plt.title('Accuracy vs Statistical Parity Difference (UCI Adult)')
plt.grid(True)
plt.legend(loc='upper left', ncol=2, fontsize="xx-small")
plt.savefig('adult_compare.pdf')
