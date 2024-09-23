#!/bin/bash

python test.py \
    --gpu_id 0 \
    --base_model_name "seg_network" \
    --data_path /home/yyk/datasets/mvtec_anomaly_detection/ \
    --checkpoint_path ./checkpoints/grid/
