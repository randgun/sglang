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

# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export INF_NAN_MODE_FORCE_DISABLE=1
export SGLANG_SET_CPU_AFFINITY=1

export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
# export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=2000

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
# First-pass W4A8 validation.  Set STRICT=1 after the audit report contains
# no unexpected draft tensors; strict mode aborts instead of running with an
# incompletely loaded draft model.
export SGLANG_DSPARK_QUANT_AUDIT=1
export SGLANG_DSPARK_QUANT_AUDIT_STRICT=0

MODEL_PATH=/home/weights/DeepSeek-V4-Flash-DSpark-w4a8-test

LOCAL_IP="0.0.0.0"
echo "Launch complete W4A8 target + bundled W4A8 DSpark on ${LOCAL_IP}"
export LD_LIBRARY_PATH=/home/kelon/code/vllm-ascend/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/kelon/code/vllm-ascend/build/:$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=/home/kelon/code/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer:$ASCEND_CUSTOM_OPP_PATH
export SGLANG_DSPARK_VLLM_ASCEND_SO=/home/kelon/code/vllm-ascend/build/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256

# export ASCEND_PROCESS_LOG_PATH=./aic_err_info/logs
# export ASCEND_WORK_PATH=./aic_err_info
# export NPU_COLLECT_PATH=./aic_err_info
# export ASCEND_GLOBAL_LOG_LEVEL=3
# export HCCL_ENTRY_LOG_ENABLE=1

# export ASCEND_LAUNCH_BLOCKING=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --quantization modelslim \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path "${MODEL_PATH}" \
    --speculative-draft-model-quantization modelslim \
    --speculative-draft-attention-backend ascend \
    --speculative-num-draft-tokens 6 \
    --page-size 128 \
    --tp-size 16 \
    --trust-remote-code \
    --device npu \
    --attention-backend dsv4 \
    --watchdog-timeout 9000 \
    --host "${LOCAL_IP}" \
    --port 30008 \
    --nnodes 1 \
    --mem-fraction-static 0.5 \
    --prefill-max-requests 16 \
    --max-prefill-tokens 8000 \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --max-running-requests 16 \
    --dp-size 16 \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --moe-a2a-backend deepep \
    --speculative-moe-a2a-backend deepep \
    --deepep-mode auto \
    --kv-cache-dtype bfloat16 \
    --cuda-graph-max-bs-decode 16  \
    --skip-server-warmup \
    --disable-piecewise-cuda-graph