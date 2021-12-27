from model import ProtectedAttributeClassifier

import torch
import pickle
import random
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.preprocessing import MinMaxScaler
from sklego.metrics import equal_opportunity_score


from tqdm import tqdm

gpu = 0
device = torch.device(gpu if torch.cuda.is_available() else "cpu")

model = ProtectedAttributeClassifier()

MODEL_NAME = f"model-{int(time.time())}"
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss()
scheduler = ExponentialLR(optimizer, gamma=0.90)

print("init done")

whole_data_raw = pd.read_csv('adult', sep=', ', engine='python')
best_dec_acc = 0
best_train_acc = 0

shuffled = whole_data_raw.sample(frac=1, random_state=42).reset_index(drop=True)
df_train_raw = shuffled[0:int(len(shuffled) * 0.8)]
df_train_raw.reset_index(inplace=True)
df_dev_raw = shuffled[int(len(shuffled) * 0.8):int(len(shuffled) * 0.8 + len(shuffled) * 0.1)]
df_dev_raw.reset_index(inplace=True)
df_test_raw = shuffled[int(len(shuffled) * 0.8 + len(shuffled) * 0.1):]
df_test_raw.reset_index(inplace=True)

n_train = len(df_train_raw)
n_dev = len(df_dev_raw)
n_test = len(df_test_raw)
general_n_test = len(df_test_raw)
df_raw = pd.concat([df_train_raw, df_dev_raw, df_test_raw])


def work_func(x):
    if x == 'Private':
        return 0
    elif x == 'State-gov':
        return 1
    elif x == 'Self-emp-not-inc':
        return 2
    elif x == 'Self-emp-inc':
        return 3
    elif x == 'Federal-gov':
        return 4
    elif x == 'Local-gov':
        return 5
    elif x == 'Without-pay':
        return 6


def education_func(x):
    if x == 'Masters':
        return 0
    elif x == '9th':
        return 1
    elif x == 'Some-college':
        return 2
    elif x == 'Assoc-acdm':
        return 3
    elif x == 'HS-grad':
        return 4
    elif x == '11th':
        return 5
    elif x == 'Bachelors':
        return 6
    elif x == '10th':
        return 7
    elif x == 'Assoc-voc':
        return 8
    elif x == '7th-8th':
        return 9
    elif x == '5th-6th':
        return 10
    elif x == '12th':
        return 11
    elif x == 'Doctorate':
        return 12
    elif x == 'Prof-school':
        return 13
    elif x == 'Preschool':
        return 14


def marital_func(x):

    if x == 'Married-civ-spouse':
        return 0
    elif x == 'Never-married':
        return 1
    elif x == 'Widowed':
        return 2
    elif x == 'Divorced':
        return 3
    elif x == 'Separated':
        return 4
    elif x == 'Married-spouse-absent':
        return 5
    elif x == 'Married-AF-spouse':
        return 6


def occupation_func(x):

    if x == 'Sales':
        return 0
    elif x == 'Farming-fishing':
        return 1
    elif x == 'Transport-moving':
        return 2
    elif x == 'Exec-managerial':
        return 3
    elif x == 'Craft-repair':
        return 4
    elif x == 'Prof-specialty':
        return 5
    elif x == 'Other-service':
        return 6
    elif x == 'Tech-support':
        return 7
    elif x == 'Adm-clerical':
        return 8
    elif x == 'Machine-op-inspct':
        return 9
    elif x == 'Handlers-cleaners':
        return 10
    elif x == 'Protective-serv':
        return 11
    elif x == 'Priv-house-serv':
        return 12
    elif x == 'Armed-Forces':
        return 13


def relationship_func(x):
    if x == 'Husband':
        return 0
    elif x == 'Other-relative':
        return 1
    elif x == 'Wife':
        return 2
    elif x == 'Unmarried':
        return 3
    elif x == 'Own-child':
        return 4
    elif x == 'Not-in-family':
        return 5


def race_func(x):

    if x == 'White':
        return 0
    elif x == 'Black':
        return 1
    elif x == 'Other':
        return 2
    elif x == 'Asian-Pac-Islander':
        return 3
    elif x == 'Amer-Indian-Eskimo':
        return 4


def sex_func(x):

    if x == 'Male':
        return 0
    elif x == 'Female':
        return 1


def country_func(x):

    if x == 'France':
        return 0
    elif x == 'United-States':
        return 1
    elif x == 'Germany':
        return 2
    elif x == 'Mexico':
        return 3
    elif x == 'Philippines':
        return 4
    elif x == 'Poland':
        return 5
    elif x == 'Cuba':
        return 6
    elif x == 'El-Salvador':
        return 7
    elif x == 'India':
        return 8
    elif x == 'Puerto-Rico':
        return 9
    elif x == 'Canada':
        return 10
    elif x == 'Thailand':
        return 11
    elif x == 'Vietnam':
        return 12
    elif x == 'England':
        return 13
    elif x == 'Haiti':
        return 14
    elif x == 'Italy':
        return 15
    elif x == 'Greece':
        return 16
    elif x == 'Outlying-US(Guam-USVI-etc)':
        return 17
    elif x == 'Japan':
        return 18
    elif x == 'Yugoslavia':
        return 19
    elif x == 'China':
        return 20
    elif x == 'Guatemala':
        return 21
    elif x == 'Honduras':
        return 22
    elif x == 'Jamaica':
        return 23
    elif x == 'Peru':
        return 24
    elif x == 'Dominican-Republic':
        return 25
    elif x == 'Ireland':
        return 26
    elif x == 'Portugal':
        return 27
    elif x == 'Taiwan':
        return 28
    elif x == 'Iran':
        return 29
    elif x == 'South':
        return 30
    elif x == 'Hong':
        return 31
    elif x == 'Ecuador':
        return 32
    elif x == 'Nicaragua':
        return 33
    elif x == 'Laos':
        return 34
    elif x == 'Cambodia':
        return 35
    elif x == 'Columbia':
        return 136
    elif x == 'Scotland':
        return 37
    elif x == 'Trinadad&Tobago':
        return 38
    elif x == 'Hungary':
        return 39
    elif x == 'Holand-Netherlands':
        return 40


def y_func(x):

    if x == '<=50K':
        return 0
    elif x == '>50K':
        return 1


df_raw['workclass'] = df_raw['workclass'].apply(work_func)
df_raw['education'] = df_raw['education'].apply(education_func)
df_raw['marital-status'] = df_raw['marital-status'].apply(marital_func)
df_raw['occupation'] = df_raw['occupation'].apply(occupation_func)
df_raw['relationship'] = df_raw['relationship'].apply(relationship_func)
df_raw['race'] = df_raw['race'].apply(race_func)
df_raw['sex'] = df_raw['sex'].apply(sex_func)
df_raw['native-country'] = df_raw['native-country'].apply(country_func)
df_raw['Y'] = df_raw['Y'].apply(y_func)
df_raw.dropna(inplace=True)
scaler = MinMaxScaler()
y = df_raw['sex']
X = df_raw.drop(['sex', 'Y', 'index'], axis=1)
X = scaler.fit_transform(X)

X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X = np.array(X)
X_test = np.array(X_test)
y = np.array(y)
y_test = np.array(y_test)


def fwd_pass(x, y_l, train=False):
    if train:
        model.zero_grad()
    x = torch.Tensor(x).to(torch.float)
    out = model(x.to(device))
    out = out[0].to(torch.float)
    y_l = torch.Tensor(y_l).view(-1, 1).to(torch.float)
    y_l = y_l.to(device)
    loss = criterion(out, y_l)
    matches = [torch.round(i) == torch.round(j) for i, j in zip(out, y_l)]
    acc = matches.count(True) / len(matches)

    if train:
        loss.backward()
        optimizer.step()

    return acc, loss, out


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
    print(acc)


def train(net):
    EPOCHS = 350
    BATCH_SIZE = 500

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
                    acc, loss, _ = fwd_pass(batch_X, batch_y, train=True)

                    losses.append(loss.item())
                    accs.append(acc)
                    acc_mean = np.array(accs).mean()
                    loss_mean = np.array(losses).mean()
                    tepoch.set_postfix(loss=loss_mean, accuracy=100. * acc_mean)
                    if i % 100000 == 0:
                        test_func(model, y_test, X_test)
                        # print(f'Average Loss: {val_loss}')
                        # print(f'Average Accuracy: {val_acc}')
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


train(model)
torch.save(model.state_dict(), 'protected.pt')