#!/usr/bin/env bash

# z 20

# match degrees of freedom
python run_models.py --z_dims 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 --epochs 300 --val_f 2 --save_f 5 --log_dir ambient_v2_20 --batch 64 --max_restarts 40
python run_models.py --z_dims 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 --epochs 300 --val_f 2 --save_f 5 --log_dir ambient_v2_20 --batch 64 --max_restarts 40
python run_models.py --z_dims 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 --epochs 300 --val_f 2 --save_f 5 --log_dir ambient_v2_20 --batch 64 --max_restarts 40

# match ambient space
python run_models.py --z_dims 1,1,1,1,1,1,1,1,1,2 --epochs 300 --val_f 2 --save_f 5 --log_dir ambient_v2_20 --batch 64 --max_restarts 40
python run_models.py --z_dims 1,1,1,1,1,1,1,1,1,2 --epochs 300 --val_f 2 --save_f 5 --log_dir ambient_v2_20 --batch 64 --max_restarts 40
python run_models.py --z_dims 1,1,1,1,1,1,1,1,1,2 --epochs 300 --val_f 2 --save_f 5 --log_dir ambient_v2_20 --batch 64 --max_restarts 64