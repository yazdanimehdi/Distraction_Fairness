import pickle
import time

import pandas as pd
import numpy as np
from fairtorch import DemographicParityLoss, EqualiedOddsLoss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklego.metrics import equal_opportunity_score

import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from metrics import demographic_parity_difference_soft
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


MODEL_NAME = f"model-{int(time.time())}"




def fwd_pass(x, y_l, k):
    model.zero_grad()
    x = torch.Tensor(x).to(torch.float)
    y_l = torch.Tensor(y_l).view(-1, 1).to(torch.float)
    y_l = y_l.to(device)
    out = model(x.to(device))[0]
    out = out.to(torch.float)
    loss_dp = criterion_dp(x, out, x[:, 123], y_l)
    matches = [torch.round(i) == torch.round(j) for i, j in zip(out, y_l)]
    acc = matches.count(True) / len(matches)
    loss_acc = criterion_acc(out, y_l)
    if k % 20 == 0:
        for name, param in model.named_parameters():
            if name in ['attention.qkv_proj.weight', 'attention.qkv_proj.bias', 'attention.o_proj.weight',
                        'attention.o_proj.bias']:
                param.require_grad = True
            else:
                param.require_grad = False

        loss_dp.backward()
        optimizer_dp.step()
        out = model(x.to(device))[0]
        out = out.to(torch.float)
        loss_acc = criterion_acc(out, y_l)
    for name, param in model.named_parameters():
        if name in ['attention.qkv_proj.weight', 'attention.qkv_proj.bias', 'attention.o_proj.weight',
                    'attention.o_proj.bias']:
            param.require_grad = False
        else:
            param.require_grad = True

    # out = model(x.to(device))[0]
    # out = out.to(torch.float)

    loss_acc.backward()
    optimizer_acc.step()

    return acc, loss_acc, loss_dp.detach().numpy(), out


def sdp(x, y):
    male_and_high = [1 if (i == 1 and torch.round(j) == 1) else 0 for i, j in zip(x[:, 9], y)].count(1)
    male = [i for i in x[:, 9]].count(1)
    female_and_high = [1 if (i == 0 and torch.round(j) == 1) else 0 for i, j in zip(x[:, 9], y)].count(1)
    female = [j for j in x[:, 9]].count(0)

    p_male_high = male_and_high / male
    p_female_high = female_and_high / female

    return abs(p_male_high - p_female_high)


def test_func(model_f, y_label, X_test_f):
    y_pred = []
    y_label = torch.Tensor(y_label)
    print("Testing:")
    print("-------------------")
    with tqdm(range(0, len(X_test_f), 100)) as tepoch:
        for i in tepoch:
            with torch.no_grad():
                x = torch.Tensor(X_test_f[i: i + 100]).to(device)
                y_pred.append(model_f(x)[0].cpu())

    y_pred = torch.cat(y_pred, dim=0)
    matches = [torch.round(i) == torch.round(j) for i, j in zip(y_label, y_pred)]
    acc = matches.count(True) / len(matches)
    return acc, demographic_parity_difference_soft(y_label, X_test_f[:, 123], y_pred)


acc_dp = {}
for alpha in [50]:
    model = AttributeClassifier(model_protected)
    optimizer_acc = torch.optim.Adam(model.get_linear_parameters(), lr=1e-3)
    optimizer_dp = torch.optim.Adam(model.get_attention_parameters(), lr=1e-5)
    criterion_acc = torch.nn.BCELoss()
    scheduler_dp = ExponentialLR(optimizer_dp, gamma=0.9)
    scheduler_acc = ExponentialLR(optimizer_acc, gamma=0.9)
    criterion_dp = DemographicParityLoss(sensitive_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8], alpha=alpha)
    EPOCHS = 50
    BATCH_SIZE = 100
    test_acc = []
    test_dp = []
    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            losses = []
            accs = []
            losses_dp = []
            with tqdm(range(0, len(X), BATCH_SIZE)) as tepoch:
                for i in tepoch:
                    tepoch.set_description(f"Alpha{alpha}, Epoch {epoch + 1}")

                    batch_X = X[i: i + BATCH_SIZE]
                    batch_y = y[i: i + BATCH_SIZE]

                    acc, loss, loss_dp, _ = fwd_pass(batch_X, batch_y, i)

                    losses.append(loss.item())
                    losses_dp.append(loss_dp)
                    accs.append(acc)
                    acc_mean = np.array(accs).mean()
                    loss_mean = np.array(losses).mean()
                    loss_dp_mean = np.array(losses_dp).mean()
                    tepoch.set_postfix(loss=loss_mean, accuracy=100. * acc_mean, loss_dp=loss_dp_mean)
                    if i == 0:
                        acc, sdp = test_func(model, y_test, X_test)
                        test_acc.append(acc)
                        test_dp.append(sdp[0])
                        print(f'ACC: {acc}')
                        print(f'SDP: {sdp}')
                        f.write(
                            f"{MODEL_NAME},{epoch},{round(float(acc_mean), 2)},{round(float(loss_mean), 4)},{acc},{sdp}\n")
                # scheduler.step()
            if (epoch + 1) % 20 == 0:
                scheduler_acc.step()
                scheduler_dp.step()
            dt = time.strftime("%Y_%m_%d-%H_%M_%S")

    acc_dp[alpha] = (test_acc, test_dp)

# with open('acc_dps_health.pkl', 'wb') as fp:
#     pickle.dump(acc_dp, fp)
