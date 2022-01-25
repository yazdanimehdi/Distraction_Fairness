import argparse

from fairtorch import DemographicParityLoss

from load_health import pre_process_and_load_health
from model import AttributeClassifierAblation

import torch
import pickle
import random
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.preprocessing import MinMaxScaler
from metrics import demographic_parity_difference_soft
from load_adult import pre_process_and_load_adult
from tqdm import tqdm


def fwd_pass(x, y_l, criterion_dp, criterion_acc, optimizer_dp, optimizer_acc, model):
    model.zero_grad()
    x = torch.Tensor(x).to(torch.float)
    y_l = torch.Tensor(y_l).view(-1, 1).to(torch.float)
    y_l = y_l.to(device)
    out = model(x.to(device))
    out = out.to(torch.float)

    loss_dp = criterion_dp(y_l, out, x[:, protected])
    loss_dp.backward()
    optimizer_dp.step()

    out = model(x.to(device))
    out = out.to(torch.float)
    loss_acc = criterion_acc(out, y_l)
    loss_acc.backward()
    optimizer_acc.step()
    out = model(x.to(device))

    matches = [torch.round(i) == torch.round(j) for i, j in zip(out, y_l)]
    acc = matches.count(True) / len(matches)

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
                y_pred.append(model_f(x).cpu())

    y_pred = torch.cat(y_pred, dim=0)
    matches = [torch.round(i) == torch.round(j) for i, j in zip(y_label, y_pred)]
    acc = matches.count(True) / len(matches)
    return acc, demographic_parity_difference_soft(y_label, X_test_f[:, protected], y_pred)


acc_dp = {}


class FairLossFunc(torch.nn.Module):
    def __init__(self, eta, protected):
        super(FairLossFunc, self).__init__()
        self.protected = protected
        self.eta = eta

    def forward(self, y_label, y_pred, protected):
        losses_max = torch.Tensor([0])
        for i in self.protected:
            for j in self.protected:
                index_c1 = protected == i
                index_c2 = protected == j
                p_1 = torch.mean(y_pred[index_c1])
                p_2 = torch.mean(y_pred[index_c2])
                l = ((p_1 - p_2) ** 2)
                if losses_max.item() < l.item():
                    losses_max = l

        return losses_max


def train_model(eta, mode, data, f_layers, a_layers, f_position):
    MODEL_NAME = f"model-{int(time.time())}"
    for e in [1, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]:
        if data == 'Adult':
            X, y, X_test, y_test = pre_process_and_load_adult()
        elif data == 'Health':
            X, y, X_test, y_test = pre_process_and_load_health()
        else:
            raise NotImplementedError()

        model = AttributeClassifierAblation(dataset=data, fairness_layer_mode=mode, fairness_layers=f_layers,
                                            accuracy_layers=a_layers, fairness_layers_position=f_position)
        optimizer_acc = torch.optim.Adam(model.get_accuracy_parameters(), lr=alr)
        optimizer_dp = torch.optim.Adam(model.get_fairness_parameters(), lr=flr)
        criterion_acc = torch.nn.BCELoss()
        scheduler_dp = ExponentialLR(optimizer_dp, gamma=0.9)
        scheduler_acc = ExponentialLR(optimizer_acc, gamma=0.9)
        s_c = [0, 1, 2, 3, 4, 5, 6, 7, 8] if data == 'Health' else [0, 1]
        criterion_dp = DemographicParityLoss(sensitive_classes=s_c, alpha=e)

        test_acc = []
        test_dp = []
        with open("model.log", "a") as f:
            for epoch in range(EPOCHS):
                losses = []
                accs = []
                losses_dp = []
                with tqdm(range(0, len(X), BATCH_SIZE)) as tepoch:
                    for i in tepoch:
                        tepoch.set_description(f"Alpha{eta}, Epoch {epoch + 1}")

                        batch_X = X[i: i + BATCH_SIZE]
                        batch_y = y[i: i + BATCH_SIZE]

                        acc, loss, loss_dp, _ = fwd_pass(batch_X, batch_y, criterion_dp, criterion_acc, optimizer_dp,
                                                         optimizer_acc, model)

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

        acc_dp[eta] = (test_acc, test_dp)

    with open(f'acc_dps_{data}_{len(fairness_layers)}_{mode}.pkl', 'wb') as fp:
        pickle.dump(acc_dp, fp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help='mode of the fairness layer', default='linear')
    parser.add_argument("-e", "--eta", help="eta", default=100)
    parser.add_argument("-d", "--data", help="dataset name", default='Health')
    parser.add_argument("-fl", "--fairness_layers", nargs="+", help="Fairness Layers")
    parser.add_argument("-al", "--acc_layers", nargs="+", help="Accuracy Layers")
    parser.add_argument("-fp", "--fairness_position", help="fairness layer position", default=3)
    parser.add_argument("-dv", "--device", default="cpu")
    parser.add_argument("-ep", "--epochs", default=50)
    parser.add_argument("-flr", "--fairness_learning_rate", default=1e-5)
    parser.add_argument("-nlr", "--network_learning_rate", default=1e-3)
    parser.add_argument("-bs", "--batch_size", default=100)
    args = parser.parse_args()

    alr = args.network_learning_rate
    flr = args.fairness_learning_rate
    BATCH_SIZE = int(args.batch_size)
    EPOCHS = int(args.epochs)
    if args.device != "cpu":
        assert (torch.cuda.is_available())

    device = torch.device(args.device)
    if args.data == "Adult":
        acc_layers = (14, 64, 32, 1)
        fairness_layers = (32, 32)
        protected = 9
    elif args.data == "Health":
        acc_layers = (125, 256, 128, 1)
        fairness_layers = (128, 128)
        protected = 123
    else:
        raise NotImplementedError()
    if args.acc_layers is not None:
        acc_layers = tuple(map(lambda x: int(x), args.acc_layers))

    if args.fairness_layers is not None:
        fairness_layers = tuple(map(lambda x: int(x), args.fairness_layers))

    train_model(args.eta, args.mode, args.data, fairness_layers, acc_layers, int(args.fairness_position))
