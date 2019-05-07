import numpy as np
from numba import njit
from utils import rv_discrete


def special_rad(p, a, size=1000):
    assert 0 <= p <= 1
    xk = np.array([-1, 1,  np.sqrt(a), -np.sqrt(a)])
    pk = (p / 2, p / 2, (1 - p) / 2, (1 - p) / 2)
    custm = rv_discrete(name='special_rad', values=(xk, pk))
    return custm.rvs(size=size)


def kurtosis_estimator(x):
    mu_4 = np.mean(x**4)
    var_2 = np.var(x, ddof=1)**2
    kurtosis = mu_4 / var_2
    return kurtosis


@njit(fastmath=True, parallel=True)
def true_var(p, a):
    assert 0 <= p <= 1
    return p + a*(1 - p)


@njit(fastmath=True, parallel=True)
def kurt_special_rad(p, a):
    assert 0 <= p <= 1
    num = p + (a**2 * (1 - p))
    denom = (true_var(p, a)) ** 2
    return num / denom


@njit(fastmath=True, parallel=True)
def bagging_np(x):
    indices = np.arange(len(x))
    selected_indices = np.random.choice(indices, size=len(x), replace=True)
    return x[selected_indices]


@njit(fastmath=True, parallel=True)
def xp_2(n_trials, n, N, a):
    mse_var_array, mse_var_bag_array = np.empty(len(proba_range)), np.empty(len(proba_range))
    for idp in range(len(proba_range)):
        print('loop : #', idp)
        total_var, total_var_bag = np.empty(n_trials), np.empty(n_trials)
        for i in range(n_trials):
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

        BIAS_sq_var = (EXP_var - true_var(proba_range[idp], a)) ** 2
        BIAS_sq_var_bag = (EXP_var_bag - true_var(proba_range[idp], a)) ** 2

        VAR_VAR_bag = (len(total_var_bag) / (len(total_var_bag) - 1)) * np.var(total_var_bag)
        VAR_VAR = (len(total_var_bag) / (len(total_var_bag) - 1)) * np.var(total_var)

        MSE_VAR = VAR_VAR + BIAS_sq_var
        MSE_VAR_BAG = VAR_VAR_bag + BIAS_sq_var_bag

        mse_var_array[idp] = MSE_VAR
        mse_var_bag_array[idp] = MSE_VAR_BAG
    return mse_var_array, mse_var_bag_array


if __name__ == '__main__':
    from tqdm import tqdm
    from plot_experiment_3 import plot_kurt
    import argparse
    parser = argparse.ArgumentParser(description='TLBiLSTM network')
    # parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=10, help="num trials")
    parser.add_argument('--n', type=int, default=10, help="Sample Size")
    parser.add_argument('--N', type=int, default=1000, help="Number of bagged estimators")
    parser.add_argument('--inv_a', type=float, default=12, help="a parameter in special rademacher distribution")

    args = parser.parse_args()
    print("\n" + "Arguments are: " + "\n")
    print(args)
    a = (1 / args.inv_a)

    proba_range = np.arange(0.05, 0.95, 0.05)
    all_rads_sample = np.empty((len(proba_range), args.n_trials, args.n))
    for enum, prob in tqdm(enumerate(proba_range)):
        for samp in range(args.n_trials):
            x = special_rad(prob, a=a, size=args.n)
            all_rads_sample[enum, samp] = x

    mse_var_array_fin, mse_var_bag_array_fin = xp_2(n_trials=args.n_trials, n=args.n, N=args.N, a=a)
    kurts = [kurt_special_rad(p, a) for p in proba_range]
    assert len(kurts) == len(mse_var_array_fin) == len(mse_var_bag_array_fin)
    path = f'res_special_rad__a={a}_trials={args.n_trials}_N={args.N}_n={args.n}.txt'

    with open(path, 'w') as f:
        for kurt, mse_var, mse_var_bag in zip(kurts[1:], mse_var_array_fin[1:], mse_var_bag_array_fin[1:]):
            f.write(str(kurt) + '|' + str(mse_var) + '|' + str(mse_var_bag) + '\n')
        f.close()

    print('done')
    # plot_kurt(path)
