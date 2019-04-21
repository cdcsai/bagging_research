# from models.utils import *
import numpy as np
import argparse
import random
import os
from collections import Counter
from random import choices

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston, load_diabetes, fetch_california_housing


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


if __name__ == "__main__":
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description='TLBiLSTM network')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noise', type=int, default=0.5)
    parser.add_argument('--ds', type=str, default='cali', help="Dataset")
    parser.add_argument('--n_feats', type=int, default=2, help="num features")
    parser.add_argument('--n_average', type=int, default=500, help="Number or average")
    parser.add_argument('--n_samples', type=int, default=1000, help="NUmber of samples")

    args = parser.parse_args()
    print("\n" + "Arguments are: " + "\n")
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dico = dict()


    # Loading datasets

    if args.ds == 'boston':
        X, y = load_boston()['data'], load_boston()['target']
    elif args.ds == 'diabetes':
        X, y = load_diabetes()['data'], load_diabetes()['target']
    elif args.ds == 'cali':
        X, y = fetch_california_housing()['data'], fetch_california_housing()['target']
    else:
        X, y = make_regression(n_samples=args.n_samples, n_features=args.n_feats, noise=args.noise, random_state=args.seed)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=args.seed)
    print(len(x_train), len(x_test))

    # Without Bagging
    mean_mse_wob = []
    for i in range(args.n_average):
        predictions = []
        reg = LinearRegression()
        reg.fit(x_train, y_train)
        predictions.append(reg.predict(x_test))
        predictions = np.array(predictions)

        # Testing
        final_pred_ = np.mean(predictions, axis=0)
        assert len(final_pred_) == len(y_test)
        mse = mean_squared_error(y_test, final_pred_)
        mean_mse_wob.append(mse)
    mse_wob = np.mean(mean_mse_wob)

    for N in tqdm(range(5, 100)):
        mean_mse = []
        for i in range(args.n_average):
            predictions = []
            for tr in range(N):
                # prop = 1 / args.m
                # size_subset = int(prop * len(x_train))
                # x_train_sub, y_train_sub = x_train[tr*size_subset:(tr+1)*size_subset], y_train[tr*size_subset:(tr+1)*size_subset]
                # if special_bool(args.bagging):
                #     print('Bagging Activated')
                x_train_, y_train_ = bagging(x_train, y_train)

                assert len(x_train_) == len(y_train_)
                # else:
                #     x_train_, y_train_ = x_train_sub, y_train_sub

                reg = LinearRegression()
                reg.fit(x_train_, y_train_)
                predictions.append(reg.predict(x_test))
            predictions = np.array(predictions)

            # Testing
            final_pred_ = np.mean(predictions, axis=0)
            assert len(final_pred_) == len(y_test)
            mse = mean_squared_error(y_test, final_pred_)
            mean_mse.append(mse)

        dico[N] = np.mean(mean_mse)

    print('finish')
    with open(f'results_mse_{args.ds}|{args.noise}|{args.n_average}|{args.n_feats}|{args.n_samples}.txt', 'w') as f:
        f.write('mse_without_b |' + str(mse_wob) + '\n')
        for key, value in dico.items():
            f.write(str(key) + '|' + str(value) + '\n')
        f.close()
