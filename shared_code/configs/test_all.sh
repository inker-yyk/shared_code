#!/bin/bash

# Define environment variables
ex="your_experiment"
python_path="python"
test_ckpt_path="/home/ubuntu/hdd1/yyk/true_yyk/diffsion_few_shot_2/src/seg_checkpoints"
data_path="/home/yyk/datasets/mvtec_anomaly_detection"
ldm_ckpt_path="/home/ubuntu/hdd1/yyk/true_yyk/diffsion_few_shot_2/src/last.ckpt"
txt_path="/home/ubuntu/hdd1/yyk/true_yyk/diffsion_few_shot_2/ceshi/ans_v1.txt"
output_path="/home/ubuntu/hdd1/yyk/true_yyk/diffsion_few_shot_2/ceshi/final_output_v0"

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Construct the test command
command_test="$python_path ./seg_network/test.py \
--gpu_id 0 \
--base_model_name seg_network \
--data_path $data_path \
--checkpoint_path $test_ckpt_path \
--ldm_ckpt_path $ldm_ckpt_path \
--txt_path $txt_path \
--output_path $output_path"

echo "Executing test command..."
eval $command_test