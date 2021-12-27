import matplotlib.pyplot as plt
import numpy as np
import statistics

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
mlp_acc = [0.816]
mlp_dp = [0.57]
# accs_distraction = [0.6613321412606169, 0.788913723737148, 0.8043808672329013, 0.8112650871703174, 0.8186857398301296,
#                     0.8238712561466249, 0.8270004470272687, 0.8312025033527045, 0.8343316942333482, 0.8368350469378633,
#                     0.8395172105498435, 0.8415735359856952, 0.843808672329012, 0.8450603486812696, 0.8459544032185964,
#                     0.8464908359409924, 0.8467590523021904, 0.8480107286544479, 0.8481001341081806, 0.8474742959320518,
#                     0.8489047831917746, 0.8489047831917746, 0.8490835940992401, 0.8494412159141708, 0.8497094322753688]
# dps_distraction = [0.051632226, 0.21112299, 0.2250072, 0.23691024, 0.2462846, 0.25658238, 0.26600188, 0.27332997,
#                    0.28266117, 0.29135016, 0.2985354, 0.30637902, 0.3153242, 0.32315302, 0.33079547, 0.3394302,
#                    0.3462042, 0.35331255, 0.3610776, 0.36815894, 0.3749746, 0.41649073, 0.421835, 0.42635995,
#                    0.43125224]

accs_distraction = [0.7777380420205633, 0.7858739383102369, 0.7915958873491283, 0.7961555654894948, 0.8000894054537326, 0.8036656236030398, 0.8059901654000894, 0.8077782744747429, 0.8096557890031292, 0.8123379526151095, 0.8148413053196245, 0.8161823871256146, 0.8178810907465356, 0.8198480107286544, 0.8219937416182387, 0.8233348234242289, 0.824675905230219, 0.8256593652212785, 0.8268216361198033, 0.8277156906571301, 0.8292355833705856, 0.829950827000447, 0.8317389360751006, 0.8332588287885561, 0.8348681269557443, 0.835493965131873, 0.8362986142154671, 0.8369244523915959, 0.8373714796602593, 0.8381761287438534, 0.8384443451050514, 0.8387125614662494, 0.8398748323647742, 0.8404112650871703, 0.8415735359856952, 0.8421993741618239, 0.8423781850692892, 0.8424675905230219, 0.8421993741618239, 0.8423781850692892, 0.8424675905230219, 0.8421993741618239, 0.8425569959767546, 0.843808672329012]
dps_distraction = [(0.31606984, 0.11654608249664307), (0.32621473, 0.11938182512919109), (0.33346397, 0.12128179868062337), (0.34052116, 0.12395993868509929), (0.34701398, 0.12611765066782635), (0.35322425, 0.1282628377278646), (0.35992116, 0.13139301935831707), (0.36534253, 0.13359015782674152), (0.37001264, 0.13534560203552246), (0.37584358, 0.13790915807088217), (0.38129222, 0.14003925323486327), (0.38633323, 0.14222410519917805), (0.39111245, 0.1444738229115804), (0.3969824, 0.14683891932169596), (0.4019025, 0.148675537109375), (0.4068321, 0.1507876714070638), (0.4123234, 0.15305891036987304), (0.41626906, 0.15450468063354492), (0.4210444, 0.15607198079427084), (0.42648238, 0.1584319591522217), (0.43074018, 0.16008578936258952), (0.43526572, 0.1618451436360677), (0.43986973, 0.16339909235636393), (0.44500628, 0.16569094657897948), (0.44942743, 0.16726350784301758), (0.4545711, 0.16913251876831054), (0.4588127, 0.17068777084350586), (0.46335083, 0.17235438028971353), (0.46813205, 0.17441051801045734), (0.47350568, 0.17639513015747071), (0.4782386, 0.17796128590901691), (0.48179293, 0.17942012151082357), (0.48607194, 0.18117472330729167), (0.4900427, 0.18245162963867187), (0.4941062, 0.1843666394551595), (0.4980518, 0.18570686976114908), (0.5014268, 0.18757305145263672), (0.50653374, 0.18956365585327148), (0.5112735, 0.19084685643513996), (0.51466393, 0.19251594543457032), (0.51844853, 0.19333740870157878), (0.5227923, 0.1954473813374837), (0.5274368, 0.19799954096476238), (0.5304777, 0.19837052027384441)]
dps_distraction = [i[0] for i in dps_distraction]

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
plt.title('Accuracy vs Statistical Parity Difference (Health)')
plt.grid(True)
plt.legend(loc='upper left', ncol=2, fontsize="xx-small")
#plt.show()
plt.savefig('health_compare_2.pdf')
