#!/usr/bin/env bash

# z 5
# match degrees of freedom
python run_models.py --z_dims 1,1,1,1,1 --epochs 300 --val_f 2 --save_f 5 --log_dir ambient_v2_5 --batch 64
python run_models.py --z_dims 1,1,1,1,1 --epochs 300 --val_f 2 --save_f 5 --log_dir ambient_v2_5 --batch 64
python run_models.py --z_dims 1,1,1,1,1 --epochs 300 --val_f 2 --save_f 5 --log_dir ambient_v2_5 --batch 64

# match ambient space
python run_models.py --z_dims 1,2 --epochs 300 --val_f 2 --save_f 5 --log_dir ambient_v2_5 --batch 64
python run_models.py --z_dims 1,2 --epochs 300 --val_f 2 --save_f 5 --log_dir ambient_v2_5 --batch 64
python run_models.py --z_dims 1,2 --epochs 300 --val_f 2 --save_f 5 --log_dir ambient_v2_5 --batch 64
