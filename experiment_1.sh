#!/usr/bin/env bash

python models/bagging/LinReg/bagging_playground_regression.py --ds=cali --n_average=100 &&
python models/bagging/LinReg/bagging_playground_regression.py --ds=cali --n_average=200 &&
python models/bagging/LinReg/bagging_playground_regression.py --ds=cali --n_average=400 &&
python models/bagging/LinReg/bagging_playground_regression.py --ds=other --n_average=1000 --n_samples=1000 --n_feats=5 &&
python models/bagging/LinReg/bagging_playground_regression.py --ds=other --n_average=1000 --n_samples=1000 --n_feats=2 &&
python models/bagging/LinReg/bagging_playground_regression.py --ds=other --n_average=1000 --n_samples=1000 --n_feats=10 --noise=2
python models/bagging/LinReg/bagging_playground_regression.py --ds=other --n_average=1000 --n_samples=2000 --n_feats=10 --noise=2
python models/bagging/LinReg/bagging_playground_regression.py --ds=other --n_average=500 --n_samples=5000 --n_feats=10 --noise=0.5


