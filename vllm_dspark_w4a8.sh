#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4-Flash DSpark W4A8 correctness run for:
# https://github.com/QwertyJack/vllm-ascend/tree/qwertyjack/deepseek-v4-dspark-main
#
# This first-pass configuration intentionally keeps both the target and the
# drafter eager. The checkpoint currently has optional={}, so the PR's runtime
# QuaRot loader is not activated. This is suitable when QuaRot was folded into
# the checkpoint offline, as confirmed by the quantization producer.
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
MODEL_PATH="${MODEL_PATH:-/home/weights/DeepSeek-V4-Flash-DSpark-w4a8-test}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash-dspark-w4a8}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30004}"

# The PR's real-weight validation used TP4 + DP4 + EP on 16 NPUs. For one
# 8-NPU node, TP4 + DP2 keeps the same TP geometry. Override with TP_SIZE=8
# DP_SIZE=1 if DP initialization or memory placement is problematic.
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-1}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
# A/B switch for the draft vocabulary weights:
#   1: load mtp.0.embed.weight + mtp.2.head.weight
#   0: load the target embed.weight + head.weight (upstream shared-weight behavior)
DSPARK_USE_MTP_VOCAB_WEIGHTS="${DSPARK_USE_MTP_VOCAB_WEIGHTS:-0}"
export VLLM_ASCEND_DSPARK_USE_MTP_VOCAB_WEIGHTS="${DSPARK_USE_MTP_VOCAB_WEIGHTS}"

export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Run from the checked-out vllm-ascend PR branch so its local Python package is
# preferred over any separately installed vllm_ascend package.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

python3 - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.version import Version

try:
    vllm_version = version("vllm")
except PackageNotFoundError:
    raise SystemExit("vLLM is not installed in the active Python environment.")

try:
    import vllm_ascend
except Exception as exc:
    raise SystemExit(f"Could not import vllm_ascend: {exc}") from exc


ascend_path = Path(vllm_ascend.__file__).resolve()
print(f"vLLM version check passed: {vllm_version}")
print(f"vLLM-Ascend source: {ascend_path}")
PY

if [[ ! -f "${MODEL_PATH}/quant_model_description.json" ]]; then
  echo "Missing ModelSlim description: ${MODEL_PATH}/quant_model_description.json" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/quant_model_weights.safetensors.index.json" ]]; then
  echo "Missing quantized checkpoint index under ${MODEL_PATH}" >&2
  exit 1
fi

SPECULATIVE_CONFIG="$(
  printf '{"method":"mtp","num_speculative_tokens":%s,"enforce_eager":true}' \
    "${NUM_SPECULATIVE_TOKENS}"
)"

echo "Launching vLLM-Ascend DeepSeek V4 DSpark W4A8"
echo "  model=${MODEL_PATH}"
echo "  topology=TP${TP_SIZE}/DP${DP_SIZE}/EP"
echo "  endpoint=http://${HOST}:${PORT}"
echo "  speculative_config=${SPECULATIVE_CONFIG}"
echo "  draft MTP embed/head=${DSPARK_USE_MTP_VOCAB_WEIGHTS}"
echo "  runtime QuaRot=disabled (quant_model_description.json optional={} expected)"

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
  --no-enable-prefix-caching \
  --enforce-eager \
  --speculative-config "${SPECULATIVE_CONFIG}"