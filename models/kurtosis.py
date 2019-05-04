import numpy as np
from scipy import stats
from numba import njit


def special_rad(p, size=1000):
    xk = np.array([-1, 1,  np.sqrt(1 / 16), -np.sqrt(1 / 16)])
    pk = (p / 2, p / 2, (1 - p) / 2, (1 - p) / 2)
    custm = stats.rv_discrete(name='special_rad', values=(xk, pk))
    return custm.rvs(size=size)


def mean_estimator(sample=1000):
    x = np.random.gamma(size=sample)
    # y = (x - 1)**2
    return np.mean(x)


def y_2nd_moment_estimator(sample=1000):
    x = np.random.uniform(size=sample)
    y = (x - 0.5) ** 2
    return np.mean(y**2)


def sharpe_ratio_estimator(sample=1000):
    x = np.random.exponential(size=sample)
    y = (x - 0.2) ** 2
    sharpe_estimator = (np.mean(y) ** 2) / np.var(y, ddof=1)
    return sharpe_estimator


def kurtosis_estimator(x):
    mu_4 = np.mean(x**4)
    var_2 = np.var(x)**2
    kurtosis = mu_4 / var_2
    return kurtosis


def true_var(p):
    return (1 / 8)*(1 - p) + p


def kurt_special_rad(p):
    num = p + ((1 - p) / 64)
    denom = (p + ((1 - p) / 8)) ** 2
    return num / denom


if __name__ == '__main__':
    from tqdm import tqdm
    from random import choices
    from collections import defaultdict

    # TRUE_VAR = (1 / 3)
    # TRUE_VAR = 1
    SAMPLE_2 = 100000
    dico = defaultdict(list)
    N = 100
    n = 10

    for p in tqdm(np.arange(0.05, 0.95, 0.05)):
        total_var, total_var_bag = [], []
        for i in range(SAMPLE_2):
            x = special_rad(p, size=n)

            # # Estimator MSE var
            var_x = np.var(x, ddof=1)
            total_var.append(var_x)

            # Estimator MSE var avec bagging
            mean_bagg = []
            for i in range(N):
                bag = choices(x, k=len(x))
                var_x_bag = np.var(bag, ddof=1)
                mean_bagg.append(var_x_bag)
            var_bag = np.mean(mean_bagg)
            total_var_bag.append(var_bag)

        EXP_var = np.mean(total_var)
        EXP_var_bag = np.mean(total_var_bag)

        BIAS_sq_var = (EXP_var - true_var(p)) ** 2
        BIAS_sq_var_bag = (EXP_var_bag - true_var(p)) ** 2

        VAR_VAR_bag = np.var(total_var_bag, ddof=1)
        VAR_VAR = np.var(total_var, ddof=1)

        MSE_VAR = VAR_VAR + BIAS_sq_var
        MSE_VAR_BAG = VAR_VAR_bag + BIAS_sq_var_bag

        dico[p].append(MSE_VAR)
        dico[p].append(MSE_VAR_BAG)
    kurts = list(map(lambda p: kurt_special_rad(p), list(dico.keys())))

    with open(f'res_special_rad__1/12_trials={SAMPLE_2}_N={N}_n={n}.txt', 'w') as f:
        for i, (key, val) in enumerate(dico.items()):
            f.write(str(kurts[i]) + '|' + str(val[0]) + '|' + str(val[1]) + '\n')
        f.close()

    print('done')