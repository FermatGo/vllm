#!/bin/bash

# 源码安装vllm-ascend
# pip install --no-build-isolation -v -e .


#export VLLM_USE_MODELSCOPE=True

#pip install modelscope


# 加载昇腾 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 根据实际路径调整

# 告诉 vLLM 使用 Ascend 后端
export VLLM_TARGET_DEVICE=ascend

# 用哪些卡，3指的是第3 die
export ASCEND_RT_VISIBLE_DEVICES=13

export VLLM_LOGGING_LEVEL=INFO

export PYTHONPATH=/home/liudi/vllm:/home/liudi/vllm-ascend:$PYTHONPATH


#python example.py

# --block-size 16 不生效，最小128
# /home/data/Qwen3.6-27B-w8a8
# /home/liudi/weight/Qwen3-0.6B
vllm serve /home/liudi/weight/Qwen3-0.6B \
    --host 127.0.0.1 \
    --port 50000 \
    --served-model-name qwen \
    --gpu-memory-utilization 0.6 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --block-size 16 \
    --trust-remote-code


