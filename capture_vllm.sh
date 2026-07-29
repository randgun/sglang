#!/usr/bin/env bash
set -euo pipefail

# Capture the vLLM-Ascend DeepSeek-V4 DSpark path using the same tensor names
# and directory layout as scripts/dspark_stage_compare.py on SGLang.
#
# This defaults to TP8/DP1 because capture.sh uses TP8/DP1. Internal TP-sharded
# logits and MoE tensors are not meaningful to compare across different
# topologies.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT:-${SCRIPT_DIR}/vllm-ascend-main}"

MODEL_PATH="${MODEL_PATH:-/home/weights/DeepSeek-V4-Flash-DSpark-w4a8-test}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark-w4a8}"
DUMP_DIR="${DUMP_DIR:-/tmp/dspark_vllm}"
CAPTURE_RID="${CAPTURE_RID:-dspark-stage-compare-001}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30004}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-1}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# Full-chain comparison uses the model's real target hidden by default. Set
# FIXED_TARGET_HIDDEN=1 only when isolating the draft attention/MoE path.
FIXED_TARGET_HIDDEN="${FIXED_TARGET_HIDDEN:-0}"
FIXED_TARGET_HIDDEN_SEED="${FIXED_TARGET_HIDDEN_SEED:-20260723}"
EXIT_AFTER_DRAFT="${EXIT_AFTER_DRAFT:-0}"
EXIT_AFTER_ROUND="${EXIT_AFTER_ROUND:-1}"
MAX_CAPTURE_CALLS="${MAX_CAPTURE_CALLS:-8}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export VLLM_ASCEND_DSPARK_STAGE_CAPTURE=1
export VLLM_ASCEND_DSPARK_STAGE_CAPTURE_DIR="${DUMP_DIR}"
export VLLM_ASCEND_DSPARK_STAGE_CAPTURE_RID="${CAPTURE_RID}"
export VLLM_ASCEND_DSPARK_STAGE_CAPTURE_MAX_CALLS="${MAX_CAPTURE_CALLS}"
export VLLM_ASCEND_DSPARK_STAGE_CAPTURE_FIXED_TARGET_HIDDEN="${FIXED_TARGET_HIDDEN}"
export VLLM_ASCEND_DSPARK_STAGE_CAPTURE_FIXED_TARGET_HIDDEN_SEED="${FIXED_TARGET_HIDDEN_SEED}"
export VLLM_ASCEND_DSPARK_STAGE_CAPTURE_EXIT_AFTER_DRAFT="${EXIT_AFTER_DRAFT}"
export VLLM_ASCEND_DSPARK_STAGE_CAPTURE_EXIT_AFTER_ROUND="${EXIT_AFTER_ROUND}"

# Ensure the checked-out source, including the capture module, wins over any
# separately installed vllm-ascend wheel.
export PYTHONPATH="${VLLM_ASCEND_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SPECULATIVE_CONFIG="$(
  printf '{"method":"dspark","num_speculative_tokens":%s,"enforce_eager":true}' \
    "${NUM_SPECULATIVE_TOKENS}"
)"

echo "Launching vLLM-Ascend DSpark capture"
echo "  source=${VLLM_ASCEND_ROOT}"
echo "  model=${MODEL_PATH}"
echo "  dump=${DUMP_DIR}"
echo "  label=${CAPTURE_RID}"
echo "  topology=TP${TP_SIZE}/DP${DP_SIZE}"
echo "  fixed_target_hidden=${FIXED_TARGET_HIDDEN} seed=${FIXED_TARGET_HIDDEN_SEED}"
echo "  exit_after_draft=${EXIT_AFTER_DRAFT} exit_after_round=${EXIT_AFTER_ROUND}"
echo "  endpoint=http://${HOST}:${PORT}/v1/completions"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --data-parallel-size "${DP_SIZE}" \
  --enable-expert-parallel \
  --quantization ascend \
  --dtype bfloat16 \
  --trust-remote-code \
  --seed 1024 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --block-size 128 \
  --no-enable-prefix-caching \
  --enforce-eager \
  --speculative-config "${SPECULATIVE_CONFIG}"
