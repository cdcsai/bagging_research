import numpy as np
from numba import njit


@njit(fastmath=True, parallel=True)
def bagging_np(x):
    indices = np.arange(len(x))
    selected_indices = np.random.choice(indices, size=len(x), replace=True)
    return x[selected_indices]


def distribution(name):
    assert name in ['normal', 'uniform', 'rademacher']
    return eval()


@njit(fastmath=True, parallel=True)
def experiment_2(n_trials, N, n_samples_range, distribution='uniform'):
    distance_array = np.empty((n_samples_range,))
    TRUE_VAR = 1

    for n in range(1, n_samples_range + 1):
        total_var, total_var_bag = [], []
        for i in range(n_trials):
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
                bag = bagging_np(x)
                var_x_bag = (len(bag) / (len(bag) - 1)) * np.var(bag)
                mean_bagg.append(var_x_bag)
            var_bag = np.mean(mean_bagg)
            total_var_bag.append(var_bag)

        EXP_var = np.mean(total_var)
        EXP_var_bag = np.mean(total_var_bag)

        BIAS_sq_var = (EXP_var - TRUE_VAR) ** 2
        BIAS_sq_var_bag = (EXP_var_bag - TRUE_VAR) ** 2

        VAR_VAR_bag = (len(total_var_bag) / (len(total_var_bag) - 1)) * np.var(total_var_bag)
        VAR_VAR = (len(total_var) / (len(total_var) - 1)) * np.var(total_var)

        MSE_VAR = VAR_VAR + BIAS_sq_var
        MSE_VAR_BAG = VAR_VAR_bag + BIAS_sq_var_bag

        DIST = (MSE_VAR - MSE_VAR_BAG)

        distance_array[n] = DIST


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TLBiLSTM network')
    # parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noise', type=float, default=5)
    parser.add_argument('--ds', type=str, default='def', help="Dataset")
    parser.add_argument('--n_feats', type=int, default=5, help="num features")
    parser.add_argument('--n_average', type=int, default=50, help="Number or average")
    parser.add_argument('--n_samples', type=int, default=100, help="NUmber of samples")
    parser.add_argument('--n_train_iter', type=int, default=10, help="Number of train/test loop")
    parser.add_argument('--N', type=int, default=200, help="Number of train/test loop")
