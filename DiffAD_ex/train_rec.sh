#!/bin/bash

# 定义变量
ex="vae_922"  # 替换为你的实验名
python_path="python"  # 替换为你的python路径

base_config="configs/mvtec.yaml"
max_epochs="4000"
task="rec"

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# 定义命令
command="$python_path main.py \
--base $base_config \
-t \
--gpus 0,1,2,3 \
--ex $ex \
--max_epochs $max_epochs \
--task $task"

# 执行命令并实时打印输出    
eval $command
