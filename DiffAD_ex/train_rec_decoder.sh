#!/bin/bash

ex="ex_screw_v3_full_shot_concat_noise"
python_path="python"
ckpt_path="/home/ubuntu/hdd1/yyk/true_yyk/diffsion_few_shot_2/logs/$ex/decoder/checkpoints"
log_path="/home/ubuntu/hdd1/yyk/true_yyk/diffsion_few_shot_2/logs/$ex/decoder/log"
ckpt_load_path="/home/ubuntu/hdd1/yyk/true_yyk/diffsion_few_shot_2/logs/$ex/rec/checkpoints/last.ckpt"

export CUDA_VISIBLE_DEVICES=1,3  # 指定 GPU 设备

num_gpus=2  # 修改为实际的 GPU 数量

command="$python_path -m torch.distributed.launch \
--nproc_per_node=$num_gpus \
--nnodes=1 \
--node_rank=0 \
./rec_network/train_ae_decoder.py \
--lr 0.001 \
--bs 20 \
--epochs 800 \
--data_path /home/ubuntu/hdd1/yyk/ad_dataset/mvtec_anomaly_detection/ \
--anomaly_source_path /home/ubuntu/hdd1/yyk/true_yyk/diffsion_few_shot_2/datasets/dtd/images \
--checkpoint_path $ckpt_path \
--log_path $log_path \
--ckpt_load_path $ckpt_load_path"

# 执行命令
eval $command