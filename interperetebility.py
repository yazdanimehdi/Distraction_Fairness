import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model import ProtectedAttributeClassifier
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
model_protected = ProtectedAttributeClassifier()
model_protected.load_state_dict(torch.load('protected.pt', map_location=torch.device('cpu')))
model_protected.to('cpu')
a = []
for name, item in model_protected.named_parameters():
    if 'linear_layer' in name and 'bias' not in name:
        a.append(item.detach().numpy())

atts = np.load('adult_att.npy').squeeze(axis=0)

a = np.matmul(a[0].T, a[1].T)
alpha = np.zeros((32, 13))
for i in range(atts.shape[0]):
    alpha[i] = np.matmul(a, atts[i])

plt.figure(dpi=1200)
# cols = ['CT', 'age_MISS', 'sexMISS', 'no_Claims', 'no_Providers', 'no_Vendors', 'no_PCPs', 'no_PlaceSvcs', 'no_Specialities', 'no_PCG', 'no_PG', 'PayDelay_max', 'PayDelay_min', 'PayDelay_ave', 'PayDelay_stdev', 'LOS_max', 'LOS_min', 'LOS_ave', 'LOS_stdev', 'L_T_U', 'L_T_S', 'L_T_K', 'dsfs_max', 'dsfs_min', 'dsfs_range', 'dsfs_ave', 'dsfs_stdev', 'pcg1', 'pcg2', 'pcg3', 'pcg4', 'pcg5', 'pcg6', 'pcg7', 'pcg8', 'pcg9', 'pcg10', 'pcg11', 'pcg12', 'pcg13', 'pcg14', 'pcg15', 'pcg16', 'pcg17', 'pcg18', 'pcg19', 'pcg20', 'pcg21', 'pcg22', 'pcg23', 'pcg24', 'pcg25', 'pcg26', 'pcg27', 'pcg28', 'pcg29', 'pcg30', 'pcg31', 'pcg32', 'pcg33', 'pcg34', 'pcg35', 'pcg36', 'pcg37', 'pcg38', 'pcg39', 'pcg40', 'pcg41', 'pcg42', 'pcg43', 'pcg44', 'pcg45', 'pcg46', 'sp1', 'sp2', 'sp3', 'sp4', 'sp5', 'sp6', 'sp7', 'sp8', 'sp9', 'sp10', 'sp11', 'sp12', 'sp13', 'pg1', 'pg2', 'pg3', 'pg4', 'pg5', 'pg6', 'pg7', 'pg8', 'pg9', 'pg10', 'pg11', 'pg12', 'pg13', 'pg14', 'pg15', 'pg16', 'pg17', 'pg18', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7', 'ps8', 'ps9', 'drugCount_max', 'drugCount_min', 'drugCount_ave', 'drugcount_months', 'labCount_max', 'labCount_min', 'labCount_ave', 'labcount', 'labNull', 'drugNull', 'sex']
cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
df = pd.DataFrame(alpha, columns=cols, index=[i for i in range(0, 32)])
ax = sns.heatmap(df, cmap="icefire")
ax.set_ylabel('Embeddings')
ax.set_xlabel('Input Attributes')
ax.set_title('UCI Adult')
plt.savefig('adult_heat_map.pdf')

res = np.mean(alpha, axis=0)
most_unfair = np.argwhere((-0.2 < res) & (res < 0.2))
