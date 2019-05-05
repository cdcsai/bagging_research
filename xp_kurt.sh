#!/usr/bin/env bash
python3 experiment_3.py --n_trials=10000 --n=10 --N=100 --inv_a=16 &&
python experiment_3.py --n_trials=10000 --n=10 --N=150 --inv_a=16 &&
python experiment_3.py --n_trials=10000 --n=10 --N=100 --inv_a=12
