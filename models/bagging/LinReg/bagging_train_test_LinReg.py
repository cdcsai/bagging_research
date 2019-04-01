# from models.utils import *
import numpy as np
import argparse
import random
import os
from collections import Counter
from random import choices

from sklearn.datasets import load_boston, load_diabetes, fetch_california_housing
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

    dico = dict()

    # Loading datasets
    if args.ds == 'boston':
        X, y = load_boston()['data'], load_boston()['target']
    elif args.ds == 'diabetes':
        X, y = load_diabetes()['data'], load_diabetes()['target']
    else:
        X, y = fetch_california_housing()['data'], fetch_california_housing()['target']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.99)
    print(len(x_train), len(x_test))

    for N in tqdm(range(5, 1000)):

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

        acc = mean_squared_error(y_test, final_pred_)
        dico[N] = acc

        # with open(os.path.join('/home/charles/Desktop/deep_nlp_research/models/bagging/LogReg',
        #                        'results_bagging_logreg.txt'), 'a') as f:
        #     f.write(f'{args.bagging}|{args.m}|{str(acc)}' + '\n')

    print('finish')

