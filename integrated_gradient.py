import torch
from captum.attr import IntegratedGradients
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch.nn as nn
from model import ProtectedAttributeClassifier, AttributeClassifier


def gather_labels(df):
    labels = []
    for j in range(df.shape[1]):
        if type(df[0, j]) is str:
            labels.append(np.unique(df[:, j]).tolist())
        else:
            labels.append(np.median(df[:, j]))
    return labels


raw_df = pd.read_csv("health.csv")
raw_df = raw_df[raw_df['YEAR_t'] == 'Y3']
sex = raw_df['sexMISS'] == 0
age = raw_df['age_MISS'] == 0
raw_df = raw_df.drop(['DaysInHospital', 'MemberID_t', 'YEAR_t'], axis=1)
raw_df = raw_df[sex & age]
ages = raw_df[[f'age_{i}5' for i in range(0, 9)]]
sexs = raw_df[['sexMALE', 'sexFEMALE']]
charlson = raw_df['CharlsonIndexI_max']

x = raw_df.drop(
    [f'age_{i}5' for i in range(0, 9)] + ['sexMALE', 'sexFEMALE', 'CharlsonIndexI_max',
                                          'CharlsonIndexI_min',
                                          'CharlsonIndexI_ave', 'CharlsonIndexI_range',
                                          'CharlsonIndexI_stdev',
                                          'trainset'], axis=1)
u = ages.to_numpy().argmax(axis=1)
x['age'] = u
x['sex'] = sexs.to_numpy().argmax(axis=1)
y = (charlson.to_numpy() > 0).astype(np.float32)
x = x.to_numpy()
labels = gather_labels(x)
target_labels = y.reshape(-1, 1)
scaler = MinMaxScaler()
X = x
X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X = np.array(X)
X_test = np.array(X_test)
y = np.array(y)
y_test = np.array(y_test)
gpu = 0
device = torch.device(gpu if torch.cuda.is_available() else "cpu")

model_protected = ProtectedAttributeClassifier()
model_protected.load_state_dict(torch.load('protected_health.pt', map_location=torch.device('cpu')))
model_protected.to(device)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


model_protected.final_layer = Identity()
target_class_index = 5

num_in = 124
input_a = torch.arange(0.0, num_in * 1.0, requires_grad=True).unsqueeze(0)

ig = IntegratedGradients(model_protected)
attributions, approximation_error = ig.attribute(input_a, target=1,
                                                 return_convergence_delta=True)
