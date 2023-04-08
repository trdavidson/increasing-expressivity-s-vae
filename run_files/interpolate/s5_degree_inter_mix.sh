#!/usr/bin/env bash

# s5 degree interpolation

# 1 shell
python run_models.py -zd 5 -e 300 -vf 1 -sf 5 --log_dir s5_degree -bs 100 -d static_mnist -et mlp -dt mlp
# 2 shell
python run_models.py -zd 4,1 -e 300 -vf 1 -sf 5 --log_dir s5_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 3,2 -e 300 -vf 1 -sf 5 --log_dir s5_degree -bs 100 -d static_mnist -et mlp -dt mlp
# 3 shell
python run_models.py -zd 3,1,1 -e 300 -vf 1 -sf 5 --log_dir s5_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 2,2,1 -e 300 -vf 1 -sf 5 --log_dir s5_degree -bs 100 -d static_mnist -et mlp -dt mlp
# 4 shell
python run_models.py -zd 2,1,1,1 -e 300 -vf 1 -sf 5 --log_dir s5_degree -bs 100 -d static_mnist -et mlp -dt mlp
# 5 shell
python run_models.py -zd 1,1,1,1,1 -e 300 -vf 1 -sf 5 --log_dir s5_degree -bs 100 -d static_mnist -et mlp -dt mlp