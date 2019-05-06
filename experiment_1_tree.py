import numpy as np
import argparse

from sklearn.datasets import make_regression
from sklearn.datasets import load_boston, load_diabetes, fetch_california_housing
from sklearn.tree import DecisionTreeRegressor

from numba import njit


def special_bool(boolean: str):
    if type(boolean) == bool:
        return boolean
    else:
        if boolean == "True":
            return True
        else:
            return False


@njit(fastmath=True, parallel=True)
def bagging_np(x, y):
    indices = np.arange(len(x))
    selected_indices = np.random.choice(indices, size=len(x), replace=True)
    return x[selected_indices, :], y[selected_indices]


@njit(fastmath=True)
def train_test_split_np(X, y, test_size=0.95, random_state=0):
    np.random.seed(random_state)
    train_size = 1 - test_size
    indices = np.random.permutation(X.shape[0])
    training_idx, test_idx = indices[:int(train_size*len(X))], indices[int(train_size*len(X)):]
    x_train, x_test, y_train, y_test = X[training_idx, :], X[test_idx, :], y[training_idx], y[test_idx]
    return x_train, x_test, y_train, y_test


@njit
def mean_squared_error_np(y_test, y_pred):
    mse = np.mean(((y_test - y_pred)**2))
    return mse


@njit(fastmath=True)
def LinReg_np_fit_predict(x_train, y_train, x_test):
    x_train = np.concatenate((x_train, np.ones((1, len(x_train))).reshape(-1, 1)), axis=1)
    res = np.linalg.lstsq(x_train, y_train)[0]
    coefs, intercept = res[:-1], res[-1]
    return (np.dot(x_test, coefs.reshape(-1, 1)) + intercept).reshape(-1, )



# @njit(fastmath=True, parallel=True)
def experiment_1(X, y, n_train_iter, n_average, num_N, test_size=0.95):
    array_wob = np.empty((1, n_train_iter))
    array_wb = np.empty((num_N, n_train_iter))
    rf = DecisionTreeRegressor()
    for itr_train in tqdm(range(n_train_iter)):
        print('loop #: ', itr_train)
        x_train, x_test, y_train, y_test = train_test_split_np(X, y, test_size=test_size, random_state=itr_train)
        print(len(x_train), len(x_test))
        assert len(x_train) == len(y_train) and len(x_test) == len(y_test)

        # Without Bagging

        # Fit, Predict, MSE
        rf.fit(x_train, y_train)
        pred = rf.predict(x_test)
        assert len(pred) == len(y_test)
        mse = mean_squared_error_np(y_test, pred)
        array_wob[:, itr_train] = mse

        # With Bagging

        for N in range(1, num_N + 1):
            mean_mse = np.empty(n_average)
            for j in range(n_average):
                predictions = np.empty((N, len(x_test)))
                for tr in range(N):
                    x_train_b, y_train_b = bagging_np(x_train, y_train)
                    assert len(x_train_b) == len(y_train_b)
                    rf.fit(x_train_b, y_train_b)
                    pred = rf.predict(x_test)
                    predictions[tr] = pred

                # Testing
                final_pred_ = np.array([np.mean(predictions[:, p]) for p in range(len(x_test))])
                assert len(final_pred_) == len(y_test)
                mse = mean_squared_error_np(y_test, final_pred_)
                mean_mse[j] = mse
            array_wb[N - 1, itr_train] = np.mean(mean_mse)
    return array_wob, array_wb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments Experiment 1')
    # parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noise', type=float, default=5)
    parser.add_argument('--ds', type=str, default='def', help="Dataset")
    parser.add_argument('--n_feats', type=int, default=2, help="num features")
    parser.add_argument('--n_average', type=int, default=50, help="Number or average")
    parser.add_argument('--n_samples', type=int, default=1000, help="NUmber of samples")
    parser.add_argument('--n_train_iter', type=int, default=10, help="Number of train/test loop")
    parser.add_argument('--N', type=int, default=50, help="Number of train/test loop")
    parser.add_argument('--test_size', type=float, default=0.5, help="TestSize")

    args = parser.parse_args()
    print("\n" + "Arguments are: " + "\n")
    print(args)
    import time
    from tqdm import tqdm

    # Loading datasets

    if args.ds == 'boston':
        X, y = load_boston()['data'], load_boston()['target']
    elif args.ds == 'diabetes':
        X, y = load_diabetes()['data'], load_diabetes()['target']
    elif args.ds == 'cali':
        X, y = fetch_california_housing()['data'], fetch_california_housing()['target']
    else:
        X, y = make_regression(n_samples=args.n_samples, n_features=args.n_feats, noise=args.noise, random_state=0)
    start = time.time()
    array_wob, array_wb = experiment_1(X=X, y=y, n_train_iter=args.n_train_iter, n_average=args.n_average,
                                       num_N=args.N, test_size=args.test_size)
    end = time.time()
    print(str(end - start) + ' seconds')
    print('finish')
    with open(f'results_mse_{args.ds}|{args.noise}|{args.n_average}|{args.n_feats}|{args.n_samples}_'
              f'decision_tree_test_size={args.test_size}.txt', 'w') as f:
        f.write(str(0) + '|' + str(np.mean(array_wob)) + '\n')
        for i, el in zip(range(1, len(array_wb)), array_wb):
            f.write(str(i) + '|' + str(np.mean(el)) + '\n')
        f.close()
