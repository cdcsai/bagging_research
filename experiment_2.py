import numpy as np
from numba import njit
from scipy.stats import semicircular


@njit(fastmath=True, parallel=True)
def bagging_np(x):
    indices = np.arange(len(x))
    selected_indices = np.random.choice(indices, size=len(x), replace=True)
    return x[selected_indices]

#
# def distribution(name):
#     assert name in ['normal', 'uniform', 'rademacher']
#     return eval()


@njit(fastmath=True, parallel=True)
def experiment_2(n_trials, N, n_samples_range):
    distance_array = np.empty((n_samples_range + 1,))
    TRUE_VAR = 1

    for n in range(2, n_samples_range + 1):
        print('Count epoch #: ', n)
        total_var, total_var_bag = np.empty(n_trials,), np.empty(n_trials,)
        for i in range(n_trials):
            if i % 1000 == 0:
                print('Count trial #: ', i)
            # x = np.random.normal(0, 1, size=n)
            # x = np.random.uniform(-1, 1, size=n)
            x = semicircular.rvs(size=n)
            # rademacher
            # x = 2 * bernouilli - 1

            # # Estimator MSE var
            var_x = (len(x) / (len(x) - 1)) * np.var(x)
            total_var[i] = var_x

            # Estimator MSE var avec bagging
            mean_bagg = np.empty(N,)
            for j in range(N):
                bag = bagging_np(x)
                var_x_bag = (len(bag) / (len(bag) - 1)) * np.var(bag)
                mean_bagg[j] = var_x_bag
            var_bag = np.mean(mean_bagg)
            total_var_bag[i] = var_bag

        EXP_var = np.mean(total_var)
        EXP_var_bag = np.mean(total_var_bag)

        BIAS_sq_var = (EXP_var - TRUE_VAR) ** 2
        BIAS_sq_var_bag = (EXP_var_bag - TRUE_VAR) ** 2

        VAR_VAR_bag = (len(total_var_bag) / (len(total_var_bag) - 1)) * np.var(total_var_bag)
        VAR_VAR = (len(total_var) / (len(total_var) - 1)) * np.var(total_var)

        MSE_VAR = VAR_VAR + BIAS_sq_var
        MSE_VAR_BAG = VAR_VAR_bag + BIAS_sq_var_bag

        DIST = (MSE_VAR_BAG - MSE_VAR)

        distance_array[n] = DIST
    return distance_array

# @njit(fastmath=True, parallel=True)


def experiment_2_biased(n_trials, N, n_samples_range):
    distance_array = np.empty((n_samples_range + 1,))
    TRUE_VAR = (1 / 4)

    for n in range(2, n_samples_range + 1):
        print('Count epoch #: ', n)
        total_var, total_var_bag = np.empty(n_trials, ), np.empty(n_trials, )
        for i in range(n_trials):
            # if i % 1000 == 0:
            # print('Count trial #: ', i)
            # x = np.random.normal(0, 1, size=n)
            # x = np.random.uniform(-1, 1, size=n)
            x = semicircular.rvs(size=n)
            # rademacher
            # x = 2 * bernouilli - 1

            # # Estimator MSE var
            var_x = np.var(x)
            total_var[i] = var_x

            # Estimator MSE var avec bagging
            mean_bagg = np.empty(N, )
            for j in range(N):
                bag = bagging_np(x)
                var_x_bag = np.var(bag)
                mean_bagg[j] = var_x_bag
            var_bag = np.mean(mean_bagg)
            total_var_bag[i] = var_bag

        EXP_var = np.mean(total_var)
        EXP_var_bag = np.mean(total_var_bag)

        BIAS_sq_var = (EXP_var - TRUE_VAR) ** 2
        BIAS_sq_var_bag = (EXP_var_bag - TRUE_VAR) ** 2

        VAR_VAR_bag = np.var(total_var_bag)
        VAR_VAR = np.var(total_var)

        MSE_VAR = VAR_VAR + BIAS_sq_var
        MSE_VAR_BAG = VAR_VAR_bag + BIAS_sq_var_bag

        DIST = (MSE_VAR_BAG - MSE_VAR)

        distance_array[n] = DIST
    return distance_array


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TLBiLSTM network')
    # parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ds', type=str, default='def', help="Dataset")
    parser.add_argument('--n_trials', type=int, default=50, help="Number or average")
    parser.add_argument('--n_samples', type=int, default=100, help="Number of samples")
    parser.add_argument('--N', type=int, default=200, help="Number of train/test loop")

    args = parser.parse_args()
    print("\n" + "Arguments are: " + "\n")
    print(args)

    # distance_arr = experiment_2(n_trials=50000, N=50, n_samples_range=50)
    # with open('distance_array.txt', 'w') as f:
    #     for el in distance_arr[2:]:
    #         f.write(str(el) + '\n')
    distance_arr_semi_circ = experiment_2_biased(n_trials=args.n_trials, N=args.N, n_samples_range=args.n_samples)
    with open(f'distance_array_trials={args.n_trials}_N={args.N}_n={args.n_samples}.txt', 'w') as f:
        for el in distance_arr_semi_circ[2:]:
            f.write(str(el) + '\n')