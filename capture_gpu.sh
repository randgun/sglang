#!/bin/bash

# Capture the three DeepSeek-V4 DSpark draft stages on 4x NVIDIA H20.
# The official bundled FP8 + FP4 checkpoint supplies both the target model
# and the three-stage DSpark draft.

set -euo pipefail

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

############################
# DSpark correctness setup
############################

export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_PREP_IN_CUDA_GRAPH=0
export SGLANG_DSPARK_FAST_KERNEL=0
export SGLANG_DSPARK_FAST_SAMPLING=0
export SGLANG_DSPARK_ENABLE_MULTI_STREAM=0

# Keep the three-stage comparison on the portable MHC path.  The optimized
# TileLang prenorm prewarmer treats chunked_prefill_size=-1 as a tensor size
# in the current code and fails before serving starts.
export SGLANG_DSV4_MHC_PREWARM=0
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0

# Use the installed DeepGEMM paged indexer.  The Torch fallback currently
# receives a two-dimensional c4_seq_lens tensor from this DSV4 call site but
# requires a one-dimensional tensor.
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0
export SGLANG_OPT_DSV4_NONPAGED_INDEXER=0

# Compile only shapes reached by this short correctness request.  Full
# precompile enumerates M=1..65536 when chunked prefill is disabled.
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_DG_CACHE_DIR="${SGLANG_DG_CACHE_DIR:-/root/.cache/deep_gemm}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-/root/.cache/tvm-ffi}"

# The official DeepSeek-V4-Flash-DSpark checkpoint stores routed experts as
# packed E2M1 FP4.  Keep them packed on H20; expanding them to FP8 roughly
# doubles expert-weight memory.
export SGLANG_DSV4_FP4_EXPERTS=1
unset SGLANG_DSV4_FP4_DEQUANT

# Match the NPU comparison run's unfused q_a / kv projection layout.  This
# affects execution layout but not the checkpoint tensor names.
export SGLANG_OPT_FUSE_WQA_WKV=0

MODEL_PATH="${MODEL_PATH:-/data/Weights/DeepSeek-V4-Flash-DSpark}"
DUMP_DIR="${DUMP_DIR:-/tmp/dspark_gpu}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30003}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.70}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Model config not found: ${MODEL_PATH}/config.json" >&2
    echo "Set MODEL_PATH to the official DeepSeek-V4-Flash-DSpark directory." >&2
    exit 1
fi

if [[ ! -f scripts/dspark_stage_compare.py ]]; then
    echo "Run this script from the SGLang repository root." >&2
    echo "Missing: scripts/dspark_stage_compare.py" >&2
    exit 1
fi

mkdir -p "${DUMP_DIR}" "${SGLANG_DG_CACHE_DIR}" "${TVM_FFI_CACHE_DIR}"

echo "Launching official FP8+FP4 DeepSeek-V4 DSpark capture"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES}"
echo "Model:        ${MODEL_PATH}"
echo "Dump dir:     ${DUMP_DIR}"
echo "Listen:       ${HOST}:${PORT}"

python3 scripts/dspark_stage_compare.py capture \
    --dump-dir "${DUMP_DIR}" \
    --rid-prefix dspark-stage-compare \
    --max-calls 8 \
    --exit-after-draft \
    --fixed-target-hidden \
    --fixed-target-hidden-seed 20260723 \
    -- \
    --model-path "${MODEL_PATH}" \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path "${MODEL_PATH}" \
    --attention-backend dsv4 \
    --speculative-draft-attention-backend dsv4 \
    --speculative-num-draft-tokens 6 \
    --page-size 256 \
    --tp-size 4 \
    --dp-size 1 \
    --trust-remote-code \
    --device cuda \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --prefill-max-requests 1 \
    --max-prefill-tokens 1024 \
    --max-running-requests 1 \
    --chunked-prefill-size -1 \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --moe-runner-backend marlin \
    --speculative-moe-runner-backend marlin \
    --moe-a2a-backend none \
    --speculative-moe-a2a-backend none \
    --watchdog-timeout 9000 \
    --soft-watchdog-timeout 1800 \
    --disable-cuda-graph \
    --host "${HOST}" \
    --port "${PORT}"
