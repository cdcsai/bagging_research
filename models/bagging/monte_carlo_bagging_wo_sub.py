import scipy
import numpy as np

# S(Y) = 1 / 8 if lambda=1

x = np.random.exponential(size=1000)


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


if __name__ == '__main__':
    from tqdm import tqdm
    from random import choices

    # TRUE_VAR = (1 / 3)
    TRUE_VAR = 1
    SAMPLE_2 = 10000
    dico = dict()
    count = 0
    N = 50

    for n in tqdm(range(2, 50)):
        total_var, total_var_bag = [], []
        for i in range(SAMPLE_2):
            # x = np.random.normal(0, 1, size=n)
            # x = np.random.uniform(-1, 1, size=n)
            bernouilli = np.random.binomial(1, 0.5, size=n)
            # rademacher
            x = 2 * bernouilli - 1

            # # Estimator MSE var
            var_x = np.var(x, ddof=1)
            total_var.append(var_x)

            # Estimator MSE var avec bagging
            mean_bagg = []
            for i in range(N):
                bag = choices(x, k=n)
                var_x_bag = np.var(bag, ddof=1)
                mean_bagg.append(var_x_bag)
            var_bag = np.mean(mean_bagg)
            total_var_bag.append(var_bag)

        EXP_var = np.mean(total_var)
        EXP_var_bag = np.mean(total_var_bag)
        # RES_bag = EXP_var_bag / TRUE_VAR
        # dico[count].append(N)
        # dico[count].append(n)
        # dico[count].append(RES_bag)

        BIAS_sq_var = (EXP_var - TRUE_VAR) ** 2
        BIAS_sq_var_bag = (EXP_var_bag - TRUE_VAR) ** 2

        VAR_VAR_bag = np.var(total_var_bag, ddof=1)
        VAR_VAR = np.var(total_var, ddof=1)

        MSE_VAR = VAR_VAR + BIAS_sq_var
        MSE_VAR_BAG = VAR_VAR_bag + BIAS_sq_var_bag

        DIST = (MSE_VAR - MSE_VAR_BAG)

        count += 1
        dico[n] = DIST

    x = 0