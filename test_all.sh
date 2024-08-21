#!/bin/bash

# Define environment variables
ex="img_feature_817"
python_path="/home/yyk/miniconda3/envs/DiffAD/bin/python"
test_ckpt_path="/hdd2/yyk/DiffAD-main_4/logs/$ex/seg/checkpoints"
data_path="/home/yyk/datasets/mvtec_anomaly_detection"
ldm_ckpt_path="/hdd2/yyk/DiffAD-main_4/logs/$ex/rec/checkpoints/last.ckpt"
txt_path="/hdd2/yyk/DiffAD-main_4/logs/$ex/ans_v3.txt"
output_path="/hdd2/yyk/DiffAD-main_4/logs/$ex/final_output_v3"

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Construct the test command
command_test="$python_path ./seg_network/test_v1.py \
--gpu_id 1 \
--base_model_name seg_network \
--data_path $data_path \
--checkpoint_path $test_ckpt_path \
--ldm_ckpt_path $ldm_ckpt_path \
--txt_path $txt_path \
--output_path $output_path"

echo "Executing test command..."
eval $command_test
