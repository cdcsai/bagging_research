#!/usr/bin/env bash
python bagging_playground_regression.py --noise=0.5 --n_feats=2 --n_average=500 --n_samples=1000 --n_train_iter=10
python bagging_playground_regression.py --noise=0.5 --n_feats=5 --n_average=500 --n_samples=1000 --n_train_iter=10
python bagging_playground_regression.py --noise=2 --n_feats=10 --n_average=500 --n_samples=1000 --n_train_iter=10
python bagging_playground_regression.py --noise=2 --n_feats=10 --n_average=500 --n_samples=2000 --n_train_iter=10