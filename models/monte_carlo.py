import scipy
import numpy as np

# S(Y) = 1 / 8 if lambda=1

x = np.random.exponential(size=1000)


def y_mean_estimator(sample=1000):
    x = np.random.exponential(size=sample)
    y = (x - 1)**2
    return np.mean(y)


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
    import itertools
    from collections import defaultdict

    dico = defaultdict(list)
    with open('results_est_var_2.txt', 'a') as f:
        f.write('Nb of samples distribution|Nb of samples to var|m|mse_var_2|mse_var|mse_var_2 < mse_var' + '\n')

    for sample_1, sample_2 in itertools.product([100, 500, 1000], [100, 500, 1000]):

        for m in [4, 10, 20]:

            mse_var_2, mse_var = [], []
            for j in tqdm(range(10)):
                total_var, total_var_2 = [], []
                for i in range(sample_2):
                    x = np.random.exponential(size=sample_1)

                    # Estimator MSE var
                    var_x = np.var(x, ddof=1)
                    total_var.append(var_x)

                    # Estimator MSE var_2
                    x_splitted = np.split(x, m)
                    np.random.shuffle(x_splitted)

                    mean_subset = []
                    for el in x_splitted:
                        var_2_x = np.var(el, ddof=1)
                        mean_subset.append(var_2_x)
                    var_2 = np.mean(mean_subset)
                    total_var_2.append(var_2)

                MSE_VAR_2 = np.var(total_var_2, ddof=1)
                MSE_VAR = np.var(total_var, ddof=1)
                mse_var.append(MSE_VAR)
                mse_var_2.append(MSE_VAR_2)

            with open('results_est_var_2.txt', 'a') as f:
                f.write(f'{str(sample_1)}|{str(sample_2)}|{str(m)}|{str(np.mean(mse_var_2))}|'
                        f'{str(np.mean(mse_var))}|{str(mse_var_2 < mse_var)}' + "\n")