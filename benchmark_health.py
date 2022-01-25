import itertools
import math

import fairtorch
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from fairtorch import DemographicParityLoss

from metrics import demographic_parity_difference_soft


class AC(nn.Module):
    def __init__(self):
        super(AC, self).__init__()
        linear_list = [125, 256, 128, 1]
        self.linear_layers = nn.ModuleList()
        for i in range(len(linear_list) - 2):
            self.linear_layers.append(nn.Linear(linear_list[i], linear_list[i+1]))

        self.final_layer = nn.Linear(linear_list[-2], linear_list[-1])

    def forward(self, x):
        for layer in self.linear_layers:
            x = F.relu(layer(x))

        return torch.sigmoid(self.final_layer(x))

import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from model import ProtectedAttributeClassifier



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

model = AC()

MODEL_NAME = f"model-{int(time.time())}"
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.BCELoss()
scheduler = ExponentialLR(optimizer, gamma=0.90)
criterion_dp = DemographicParityLoss(sensitive_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8], alpha=100)


def fwd_pass(x, y_l, train=False):
    if train:
        model.zero_grad()
    x = torch.Tensor(x).to(torch.float)
    out = model(x.to(device))
    out = out.to(torch.float)
    y_l = torch.Tensor(y_l).view(-1, 1).to(device)
    loss = criterion(out, y_l)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(out, y_l)]
    acc = matches.count(True) / len(matches)

    if train:
        loss.backward()
        optimizer.step()
    return acc, loss, criterion_dp(x, out, x[:, 123], y_l).detach().numpy()



def test_func(model_f, y_label, X_test_f):
    y_pred = []
    y_label = torch.Tensor(y_label)
    print("Testing:")
    print("-------------------")
    with tqdm(range(0, len(X_test_f), 100)) as tepoch:
        for i in tepoch:
            with torch.no_grad():
                x = torch.Tensor(X_test_f[i: i + 100]).to(device)
                y_pred.append(model_f(x).cpu())

    y_pred = torch.cat(y_pred, dim=0)
    matches = [torch.round(i) == torch.round(j) for i, j in zip(y_label, y_pred)]
    acc = matches.count(True) / len(matches)
    return acc, demographic_parity_difference_soft(y_label, X_test_f[:, 9], y_pred)


def train(net):
    EPOCHS = 50
    BATCH_SIZE = 100
    losses_a = []
    dps = []
    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            losses = []
            accs = []
            with tqdm(range(0, len(X), BATCH_SIZE)) as tepoch:
                for i in tepoch:
                    tepoch.set_description(f"Epoch {epoch + 1}")
                    try:

                        batch_X = X[i: i+BATCH_SIZE]
                        batch_y = y[i: i+BATCH_SIZE]
                    except:
                        continue
                    acc, loss, dp = fwd_pass(batch_X, batch_y, train=True)

                    losses.append(loss.item())
                    accs.append(acc)
                    losses_a.append(loss.item())
                    dps.append(dp)
                    acc_mean = np.array(accs).mean()
                    loss_mean = np.array(losses).mean()
                    tepoch.set_postfix(loss=loss_mean, accuracy=100. * acc_mean)
                    if i % 100000 == 0:
                        acc, sdp = test_func(model, y_test, X_test)
                        print(f'ACC: {acc}')
                        print(f'SDP: {sdp}')
                        f.write(
                            f"{MODEL_NAME},{round(time.time(), 3)},{round(float(acc), 2)},{round(float(loss), 4)}\n")
                #scheduler.step()
            print(f'Average Loss: {np.array(losses).mean()}')
            print(f'Average Accuracy: {np.array(accs).mean()}')
            dt = time.strftime("%Y_%m_%d-%H_%M_%S")
            fn = "without_batching" + str(dt) + str("-") + \
                 str(epoch) + "_checkpoint.pt"
            info_dict = {
                'epoch': epoch,
                'net_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }
            #torch.save(info_dict, fn)
    return losses_a, dps


losses, dps = train(model)
