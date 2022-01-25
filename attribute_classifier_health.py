import pickle
import time

import numpy as np
from fairtorch import DemographicParityLoss

import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from load_health import pre_process_and_load_health
from metrics import demographic_parity_difference_soft
from model import ProtectedAttributeClassifier, AttributeClassifier


X, y, X_test, y_test = pre_process_and_load_health()
gpu = 0
device = torch.device(gpu if torch.cuda.is_available() else "cpu")

model_protected = ProtectedAttributeClassifier()
model_protected.load_state_dict(torch.load('protected_health.pt', map_location=torch.device('cpu')))
model_protected.to(device)


MODEL_NAME = f"model-{int(time.time())}"




def fwd_pass(x, y_l):
    model.zero_grad()
    x = torch.Tensor(x).to(torch.float)
    y_l = torch.Tensor(y_l).view(-1, 1).to(torch.float)
    y_l = y_l.to(device)
    for name, param in model.named_parameters():
        if name in ['attention.qkv_proj.weight', 'attention.qkv_proj.bias', 'attention.o_proj.weight',
                    'attention.o_proj.bias']:
            param.requires_grad = True
        else:
            param.requires_grad = False
    out = model(x.to(device))[0]
    out = out.to(torch.float)
    loss_dp = criterion_dp(x, out, x[:, 123], y_l)
    loss_dp.backward()
    optimizer_dp.step()

    for name, param in model.named_parameters():
        if name in ['attention.qkv_proj.weight', 'attention.qkv_proj.bias', 'attention.o_proj.weight',
                    'attention.o_proj.bias']:
            param.requires_grad = False
        else:
            param.requires_grad = True

    out = model(x.to(device))[0]
    out = out.to(torch.float)
    matches = [torch.round(i) == torch.round(j) for i, j in zip(out, y_l)]
    acc = matches.count(True) / len(matches)
    loss_acc = criterion_acc(out, y_l)
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
for alpha in [1, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]:
    model = AttributeClassifier(model_protected)
    optimizer_acc = torch.optim.Adam(model.get_linear_parameters(), lr=1e-3)
    optimizer_dp = torch.optim.Adam(model.get_attention_parameters(), lr=1e-5)
    criterion_acc = torch.nn.BCELoss()
    scheduler_dp = ExponentialLR(optimizer_dp, gamma=0.9)
    scheduler_acc = ExponentialLR(optimizer_acc, gamma=0.9)
    criterion_dp = DemographicParityLoss(sensitive_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8], alpha=alpha)
    EPOCHS = 100
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

                    acc, loss, loss_dp, _ = fwd_pass(batch_X, batch_y)

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

with open('acc_dps_health.pkl', 'wb') as fp:
    pickle.dump(acc_dp, fp)
