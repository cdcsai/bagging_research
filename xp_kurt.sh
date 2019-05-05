#!/usr/bin/env bash
python3 experiment_3.py --n_trials=10000 --n=10 --N=100 --a=1/16 &&
python experiment_3.py --n_trials=10000 --n=10 --N=150 --a=1/16 &&
python experiment_3.py --n_trials=10000 --n=10 --N=100 --a=1/8
