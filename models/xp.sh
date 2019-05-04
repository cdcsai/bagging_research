#!/usr/bin/env bash
python bagging_xp_1_optimized.py --noise=5 --n_feats=2 --n_average=100 --n_samples=1000 --n_train_iter=10 --N=200
python bagging_xp_1_optimized.py --noise=3 --n_feats=2 --n_average=100 --n_samples=1000 --n_train_iter=10 --N=300
python bagging_xp_1_optimized.py --noise=2 --n_feats=5 --n_average=100 --n_samples=1000 --n_train_iter=10 --N=150
python bagging_xp_1_optimized.py --noise=5 --n_feats=10 --n_average=100 --n_samples=1000 --n_train_iter=10 --N=500
