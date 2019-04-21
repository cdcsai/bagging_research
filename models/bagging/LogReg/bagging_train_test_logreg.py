# from models.utils import *
import numpy as np
import argparse
import random
import os
from collections import Counter
from random import choices

from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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
    from collections import defaultdict
    parser = argparse.ArgumentParser(description='TLBiLSTM network')
    # parser.add_argument('--bagging', type=str, default=True, help="Bagging or Not")
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--N', type=int, default=1, help="Number of Models")
    parser.add_argument('--ds', type=str, default="dr", help="Dataset")
    args = parser.parse_args()
    print("\n" + "Arguments are: " + "\n")
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
    dico = dict()

    # Loading datasets
    if args.ds == 'iris':
        X, y = load_iris()['data'], load_iris()['target']
    elif args.ds == 'breast_cancer':
        X, y = load_breast_cancer()['data'], load_breast_cancer()['target']
    else:
        X, y = load_digits()['data'], load_digits()['target']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=0)
    print(len(x_train), len(x_test))

    # Without Bagging
    mean_acc_wob = []
    for j in range(100):
        predictions = []
        cls = LogisticRegression()
        cls.fit(x_train, y_train)
        predictions.append(cls.predict(x_test))
        predictions = np.array(predictions)

        # Testing
        final_pred_ = []
        for i in range(len(x_test)):
            c = Counter(predictions[:, i])
            final_pred_.append(c.most_common(1)[0][0])
        assert len(final_pred_) == len(y_test)

        acc = accuracy_score(y_test, final_pred_)
        mean_acc_wob.append(acc)
    acc_wob = np.mean(mean_acc_wob)

    for N in tqdm(range(5, 100)):
        mean_acc = []
        for j in range(100):
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

                cls = LogisticRegression()
                cls.fit(x_train_, y_train_)
                predictions.append(cls.predict(x_test))
            predictions = np.array(predictions)

            # Testing
            final_pred_ = []
            for i in range(len(x_test)):
                c = Counter(predictions[:, i])
                final_pred_.append(c.most_common(1)[0][0])
            assert len(final_pred_) == len(y_test)

            acc = accuracy_score(y_test, final_pred_)
            mean_acc.append(acc)
        dico[N] = np.mean(mean_acc)

    print('finish')

