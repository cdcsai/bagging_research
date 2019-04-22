import scipy
import numpy as np
import scipy.stats as st


#class SpecialRad(st.rv_discrete):

   # def _rvs(self, p):
       # return binom_gen._rvs(self, 1, p)


#my_cv = SpecialRad(a=0, b=1, name='my_pdf')


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


def kurtosis_estimator(dist='exponential', **kwargs):
    x = eval(f'np.random.{dist}')(**kwargs)
    mu_4 = np.mean(x**4)
    var_2 = np.var(x)**2
    kurtosis = mu_4 / var_2
    return kurtosis


if __name__ == '__main__':
    from tqdm import tqdm
    from collections import defaultdict
    from random import choices

    TRUE_VAR = 1
    SAMPLE_2 = 100000
    dico = defaultdict(list)
    count = 0
    N = 50

    for n in tqdm(range(2, 20)):
        total_var, total_var_bag = [], []
        for i in range(SAMPLE_2):
            x = np.random.normal(0, 1, size=n)

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

        DIST = MSE_VAR - MSE_VAR_BAG

        count += 1
        dico[count].append(n)
        dico[count].append(DIST)

    x = 0