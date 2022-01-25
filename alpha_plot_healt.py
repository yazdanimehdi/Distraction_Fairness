import seaborn as sns
import typing
import numpy as np
import pickle
import matplotlib.pyplot as plt


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


with open('acc_dps_health.pkl', 'rb') as fp:
    acc_dp = pickle.load(fp)

del acc_dp[3500]
del acc_dp[4000]
del acc_dp[4500]
for item in acc_dp.keys():
    a = [(i, j) for i, j in zip(acc_dp[item][0], acc_dp[item][1])]
    a.sort(key=lambda x: -x[0])
    acc_dp[item] = a

pareto_set = [[] for item in acc_dp.keys()]
for idx, item in enumerate(acc_dp.keys()):
    pareto_set[idx].append(get_pareto_front(acc_dp[item]))

keys = list(acc_dp.keys())
best_accs = [i[0][0][0] for i in pareto_set]
dps_best_acc = [i[0][0][1] for i in pareto_set]

fig = plt.figure()
ax = fig.add_subplot(111)
lns1 = ax.plot(keys, best_accs, 'kD', markerfacecolor='none', ms=6,
               markeredgecolor='red')
ax.tick_params(axis='y', labelcolor='red')
ax2 = ax.twinx()
lns2 = ax2.plot(keys, dps_best_acc, 'bs', markerfacecolor='none', ms=6, markeredgecolor='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# added these three lines
lns = lns1 + lns2

ax.grid()
ax.set_xlabel(r"$\eta$")
ax.set_ylabel("Accuracy", color='red')
ax2.set_ylabel(r"Statistical Parity Difference", color='blue')
ax2.set_ylim(0, 1)
ax.set_ylim(0.65, 0.9)
fig.savefig('different_alphas_health.pdf', bbox_inches='tight')
fig.show()
