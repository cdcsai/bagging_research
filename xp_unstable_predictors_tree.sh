#!/usr/bin/env bash
python experiment_1_tree.py --n_average=50 --n_samples=1000 --n_train_iter=10 --N=100 --n_feats=2 --test_size=0.95 --noise=0.5
python experiment_1_tree.py --n_average=50 --n_samples=1000 --n_train_iter=10 --N=100 --n_feats=2 --test_size=0.95 --noise=5
