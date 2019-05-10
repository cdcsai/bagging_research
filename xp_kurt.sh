#!/usr/bin/env bash

python experiment_3.py --n_trials=100000 --n=10 --N=50 --inv_a=8 &&
python experiment_3.py --n_trials=100000 --n=10 --N=100 --inv_a=8

