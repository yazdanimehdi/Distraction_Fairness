import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import typing


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


with open('adult_different_alpha.pkl', 'rb') as fp:
    acc_dp = pickle.load(fp)

for item in acc_dp.keys():
    a = [(i, j) for i, j in zip(acc_dp[item][0], acc_dp[item][1])]
    a.sort(key=lambda x: -x[0])
    acc_dp[item] = a

pareto_set = [acc_dp[item] for item in acc_dp.keys()]
pareto_set_all = []
for item in pareto_set:
    pareto_set_all += item

pareto_set_all = get_pareto_front(pareto_set_all)

best_accs = [i[0] for i in pareto_set_all]
dps_best_acc = [i[1] for i in pareto_set_all]

with open('acc_dps_adult_1_fairness_nn.pkl', 'rb') as fp:
    acc_dp = pickle.load(fp)


for item in acc_dp.keys():
    a = [(i, j) for i, j in zip(acc_dp[item][0], acc_dp[item][1])]
    a.sort(key=lambda x: -x[0])
    acc_dp[item] = a

pareto_set = [acc_dp[item] for item in acc_dp.keys()]
pareto_set_all = []
for item in pareto_set:
    pareto_set_all += item

pareto_set_all = get_pareto_front(pareto_set_all)

best_accs_1_nn = [i[0] for i in pareto_set_all]
dps_best_acc_1_nn = [i[1] for i in pareto_set_all]

plt.figure(dpi=1200)

plt.plot(best_accs_1_nn, dps_best_acc_1_nn, 'ro', label='1 Fairness Linear', markerfacecolor='none', ms=6, markeredgecolor='purple')
plt.plot(best_accs, dps_best_acc, 'kD', label='Distraction', markerfacecolor='none', ms=6,
         markeredgecolor='red')

plt.xlabel('Accuracy')
plt.ylabel('Statistical Parity Difference')
plt.grid(True)
plt.legend(loc='upper left', ncol=2, fontsize="xx-small")
plt.savefig('ablation_adult.pdf')
