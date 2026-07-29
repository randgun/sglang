#!/bin/bash

# Complete DeepSeek-V4 DSpark W4A8 target + bundled W4A8 draft on Ascend A3.
# Keep dsv4.sh unchanged as the known-good target-only baseline.

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset no_proxy

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

############################
# 8-card single-node setup
############################

export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export INF_NAN_MODE_FORCE_DISABLE=1
export SGLANG_SET_CPU_AFFINITY=1

export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
# export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1000

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

############################
# DeepSeek-V4 NPU fallbacks
############################

export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
unset FORCE_DRAFT_MODEL_NON_QUANT
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False

############################
# DSpark correctness-first setup
############################

export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_FAST_KERNEL=0
export SGLANG_DSPARK_FAST_SAMPLING=0
export SGLANG_DSPARK_ENABLE_MULTI_STREAM=0

MODEL_PATH=/data/weights/DeepSeek-V4-Flash-DSpark-w4a8

LOCAL_IP=$(hostname -I | awk '{print $1}')
echo "Launch complete W4A8 target + bundled W4A8 DSpark on ${LOCAL_IP}"


# export SGLANG_DSPARK_NPU_PHASE_PROBE=1
# export SGLANG_DSV4_NPU_LAYER_PROBE=1
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --quantization modelslim \
    --page-size 128 \
    --tp-size 8 \
    --trust-remote-code \
    --device npu \
    --attention-backend dsv4 \
    --watchdog-timeout 9000 \
    --host "${LOCAL_IP}" \
    --port 30002 \
    --nnodes 1 \
    --mem-fraction-static 0.69 \
    --prefill-max-requests 1 \
    --max-prefill-tokens 8000 \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --max-running-requests 1 \
    --disable-overlap-schedule \
    --dp-size 1 \
    --moe-a2a-backend deepep \
    --kv-cache-dtype bfloat16 \
    --deepep-mode auto \
    --disable-cuda-graph 
