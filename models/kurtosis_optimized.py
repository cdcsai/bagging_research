import numpy as np
from scipy import stats
from numba import njit


def special_rad(p, size=1000):
    assert 0 <= p <= 1
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


@njit(fastmath=True, parallel=True)
def true_var(p):
    assert 0 <= p <= 1
    return (1 / 8)*(1 - p) + p


def kurt_special_rad(p):
    assert 0 <= p <= 1
    num = p + ((1 - p) / 64)
    denom = (p + ((1 - p) / 8)) ** 2
    return num / denom


@njit(fastmath=True, parallel=True)
def bagging_np(x):
    indices = np.arange(len(x))
    selected_indices = np.random.choice(indices, size=len(x), replace=True)
    return x[selected_indices]


@njit(fastmath=True, parallel=True)
def xp_2():
    mse_var_array, mse_var_bag_array = np.empty(len(proba_range)), np.empty(len(proba_range))
    for idp in range(len(proba_range)):
        print('loop : #', idp)
        total_var, total_var_bag = np.empty(SAMPLE_2), np.empty(SAMPLE_2)
        for i in range(SAMPLE_2):
            x = all_rads_sample[idp, i]
            assert len(x) == n

            # # Estimator MSE var
            var_x = (len(x) / (len(x) - 1)) * np.var(x)
            total_var[i] = var_x

            # Estimator MSE var avec bagging
            mean_bagg = np.empty(N)
            for j in range(N):
                bagged_x = bagging_np(x)
                var_x_bag = (len(bagged_x) / (len(bagged_x) - 1)) * np.var(bagged_x)
                mean_bagg[j] = var_x_bag
            var_bag = np.mean(mean_bagg)
            total_var_bag[i] = var_bag

        EXP_var = np.mean(total_var)
        EXP_var_bag = np.mean(total_var_bag)

        BIAS_sq_var = (EXP_var - true_var(proba_range[idp])) ** 2
        BIAS_sq_var_bag = (EXP_var_bag - true_var(proba_range[idp])) ** 2

        VAR_VAR_bag = (len(total_var_bag) / (len(total_var_bag) - 1)) * np.var(total_var_bag)
        VAR_VAR = (len(total_var_bag) / (len(total_var_bag) - 1)) * np.var(total_var)

        MSE_VAR = VAR_VAR + BIAS_sq_var
        MSE_VAR_BAG = VAR_VAR_bag + BIAS_sq_var_bag

        mse_var_array[idp] = MSE_VAR
        mse_var_bag_array[idp] = MSE_VAR_BAG
    return mse_var_array, mse_var_bag_array


if __name__ == '__main__':
    from tqdm import tqdm

    SAMPLE_2 = 10000
    N = 100
    n = 10
    proba_range = np.arange(0.05, 0.95, 0.05)

    all_rads_sample = np.empty((len(proba_range), SAMPLE_2, n))
    for enum, prob in tqdm(enumerate(proba_range)):
        for samp in range(SAMPLE_2):
            x = special_rad(prob, size=n)
            all_rads_sample[enum, samp] = x

    mse_var_array_fin, mse_var_bag_array_fin = xp_2()
    kurts = [kurt_special_rad(p) for p in proba_range]
    assert len(kurts) == len(mse_var_array_fin) == len(mse_var_bag_array_fin)

    with open(f'res_special_rad__1over12_trials={SAMPLE_2}_N={N}_n={n}.txt', 'w') as f:
        for kurt, mse_var, mse_var_bag in zip(kurts, mse_var_array_fin, mse_var_bag_array_fin):
            f.write(str(kurt) + '|' + str(mse_var) + '|' + str(mse_var_bag) + '\n')
        f.close()

    print('done')