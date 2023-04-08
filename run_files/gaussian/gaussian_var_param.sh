#!/usr/bin/env bash

# gaussian covariance matrix variants
python run_models.py -z 1 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -cvm single -ld gaus
python run_models.py -z 2 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -cvm single -ld gaus
python run_models.py -z 5 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -cvm single -ld gaus
python run_models.py -z 10 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -cvm single -ld gaus
python run_models.py -z 20 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -cvm single -ld gaus
python run_models.py -z 40 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -cvm single -ld gaus

python run_models.py -z 1 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -ld gaus
python run_models.py -z 2 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -ld gaus
python run_models.py -z 5 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -ld gaus
python run_models.py -z 10 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -ld gaus
python run_models.py -z 20 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -ld gaus
python run_models.py -z 40 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -ld gaus

# full covariance matrix
python run_models.py -z 1 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -cvm full -ld gaus -b 1e-2 -bi 150
python run_models.py -z 2 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -cvm full -ld gaus -b 1e-2 -bi 150
python run_models.py -z 5 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -cvm full -ld gaus -b 1e-2 -bi 150
python run_models.py -z 10 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -cvm full -ld gaus -b 1e-2 -bi 150
python run_models.py -z 20 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -cvm full -ld gaus -b 1e-2 -bi 150
python run_models.py -z 40 -n normal -dis normal -ld gaussian -d static_mnist -bs 100 -cvm full -ld gaus -b 1e-2 -bi 150