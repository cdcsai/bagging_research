# from models.utils import *
import numpy as np
import argparse
import random
from random import choices

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def special_bool(boolean: str):
    if type(boolean) == bool:
        return boolean
    else:
        if boolean == "True":
            return True
        else:
            return False


def bagging(x, y):
    x_y = np.concatenate([x, y.reshape(-1, 1)], axis=1)
    bag = choices(x_y, k=len(x))
    x = [np.array(el[:-1]) for el in bag]
    y = [el[-1] for el in bag]
    return x, y

# def pointwise_stability()


if __name__ == "__main__":
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description='TLBiLSTM network')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ds', type=str, default='bost', help="Dataset")
    args = parser.parse_args()
    print("\n" + "Arguments are: " + "\n")
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Hyper-parameters
    N = 50
    mean_beta_with_bagging, mean_beta_without_bagging = [], []
    for i in range(20):
        # Loading datasets
        X, y = make_regression(n_samples=100, n_features=5, noise=0.5)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0)

        predictions_before_bagging, predictions_after_bagging = [], []

        # With Bagging
        for tr in range(N):
            x_train_, y_train_ = bagging(x_train, y_train)
            assert len(x_train_) == len(y_train_)
            reg = LinearRegression()
            reg.fit(x_train_, y_train_)
            predictions_before_bagging.append(reg.predict(x_train))
        predictions = np.array(predictions_before_bagging)
        final_pred_with_bagging = np.mean(predictions, axis=0)
        predictions_after_bagging.append(final_pred_with_bagging)

        # Without Bagging
        predictions_without_bagging = []

        reg = LinearRegression()
        reg.fit(x_train, y_train)
        final_pred_without_bagging = reg.predict(x_train)
        predictions_without_bagging.append(final_pred_without_bagging)

        # Measure the loo empirical risk with bagging
        predictions_loo_before_bagging, predictions_loo_after_bagging, predictions_loo_without_bagging = [], [], []
        for i in tqdm(range(len(x_train))):
            x_train_loo = np.delete(x_train, i, 0)
            y_train_loo = np.delete(y_train, i, 0)
            assert x_train_loo.shape[0] == 99

            # Pred Loo with bagging
            for tr in range(N):
                x_train_, y_train_ = bagging(x_train_loo, y_train_loo)
                assert len(x_train_) == len(y_train_)
                reg = LinearRegression()
                reg.fit(x_train_, y_train_)
                predictions_loo_before_bagging.append(reg.predict(x_train))
            predictions = np.array(predictions_loo_before_bagging)
            final_pred_ = np.mean(predictions, axis=0)
            predictions_loo_after_bagging.append(final_pred_)

            # Pred Loo without bagging
            reg = LinearRegression()
            reg.fit(x_train_loo, y_train_loo)
            predictions_loo_without_bagging.append(reg.predict(x_train_loo))

        # Computing pointwise stability
        pointwise_stability_with_bagging = []
        for i in range(len(x_train)):
            for pred_bag, pred_bag_loo, truth in zip(predictions_after_bagging[0],
                                                     predictions_loo_after_bagging[i],
                                                     y_train):
                # Computing the pointwise MSE
                mse_bag, mse_bag_loo = (truth - pred_bag)**2, (truth - pred_bag_loo)**2
                # Computing and appending the difference between the two of them
                pointwise_stability_with_bagging.append(np.abs(mse_bag - mse_bag_loo))
        beta_with_bagging = max(pointwise_stability_with_bagging)

        # Computing pointwise stability without bagging
        pointwise_stability_without_bagging = []
        for i in range(len(x_train)):
            for pred, pred_loo, truth in zip(predictions_without_bagging[0],
                                             predictions_loo_without_bagging[i],
                                                     y_train):
                # Computing the pointwise MSE
                mse, mse_loo = (truth - pred)**2, (truth - pred_loo)**2
                # Computing and appending the difference between the two of them
                pointwise_stability_without_bagging.append(np.abs(mse - mse_loo))
        beta_without_bagging = max(pointwise_stability_without_bagging)

        mean_beta_with_bagging.append(beta_with_bagging)
        mean_beta_without_bagging.append(beta_without_bagging)

        print('With bagging, stability is', beta_with_bagging)

        print('Without bagging, stability is', beta_without_bagging)


