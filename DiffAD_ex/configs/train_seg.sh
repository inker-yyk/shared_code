#!/bin/bash

ex="your_experiment"
python_path="python"
ckpt_path="/hdd2/yyk/DiffAD-main_2/logs/$ex/seg/checkpoints"
log_path="/hdd2/yyk/DiffAD-main_2/logs/$ex/seg/log"
ckpt_load_path="/hdd2/yyk/DiffAD-main_2/logs/$ex/rec/checkpoints/last.ckpt"

export CUDA_VISIBLE_DEVICES=0,1,2,3  # 指定 GPU 设备

num_gpus=4  # 修改为实际的 GPU 数量

command="$python_path -m torch.distributed.launch \
--nproc_per_node=$num_gpus \
--nnodes=1 \
--node_rank=0 \
./seg_network/train.py \
--lr 0.001 \
--bs 30 \
--epochs 800 \
--data_path /home/yyk/datasets/mvtec_anomaly_detection/ \
--anomaly_source_path /hdd2/yyk/DiffAD-main/datasets/dtd/images/ \
--checkpoint_path $ckpt_path \
--log_path $log_path \
--ckpt_load_path $ckpt_load_path"

# 执行命令
eval $command