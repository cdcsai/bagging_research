# from models.utils import *
import numpy as np
import argparse
import random
import os
from collections import Counter
from random import choices

from sklearn.datasets import make_classification
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

    X, y = make_classification(n_samples=10000, n_features=100, flip_y=0, n_classes=10, n_informative=10)
    # X, y = make_classification(n_samples=10000, n_features=100, flip_y=0, n_classes=10)
    # X, y = make_classification(n_samples=10000, n_features=100, flip_y=0, n_classes=20)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.99)
    print(len(x_train), len(x_test))

    for N in tqdm(range(2, 100)):

        predictions = []
        for tr in range(N):
            x_train_, y_train_ = bagging(x_train, y_train)
            assert len(x_train_) == len(y_train_)

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
        dico[N] = acc

        # with open(os.path.join('/home/charles/Desktop/deep_nlp_research/models/bagging/LogReg',
        #                        'results_bagging_logreg.txt'), 'a') as f:
        #     f.write(f'{args.bagging}|{args.m}|{str(acc)}' + '\n')

    print('finish')