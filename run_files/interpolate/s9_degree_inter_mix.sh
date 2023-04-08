#!/usr/bin/env bash

# s9 degree interpolation

# static static_mnist

# 1 shell
python run_models.py -zd 9 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
# 2 shell
python run_models.py -zd 8,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 7,2 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 6,3 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 5,4 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
# 3 shell
python run_models.py -zd 7,1,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 6,2,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 5,3,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 5,2,2 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 4,4,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 4,3,2 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 3,3,3 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
# 4 shell
python run_models.py -zd 6,1,1,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 5,2,1,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 4,3,1,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 4,2,2,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 3,3,2,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 3,2,2,2 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
# 5 shell
python run_models.py -zd 5,1,1,1,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 4,2,1,1,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 3,3,1,1,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 3,2,2,1,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 2,2,2,2,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
# 6 shell
python run_models.py -zd 4,1,1,1,1,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 3,2,1,1,1,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 2,2,2,1,1,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
# 7 shell
python run_models.py -zd 3,1,1,1,1,1,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
python run_models.py -zd 2,2,1,1,1,1,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
# 8 shell
python run_models.py -zd 2,1,1,1,1,1,1,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp
# 9 shell
python run_models.py -zd 1,1,1,1,1,1,1,1,1 -e 300 -vf 1 -sf 5 --log_dir s9_degree -bs 100 -d static_mnist -et mlp -dt mlp