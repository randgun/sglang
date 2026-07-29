export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1

export SGLANG_DSV4_FP4_EXPERTS=True
export SGLANG_DSV4_FP4_DEQUANT=1

export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

SGLANG_RAGGED_VERIFY_MODE=compact \
python3 -m sglang.launch_server \
  --model-path /data/weights/DeepSeek-V4-Flash-DSpark \
  --speculative-algorithm DSPARK \
  --attention-backend dsv4 \
  --tp-size 8 \
  --cuda-graph-max-bs-decode 4 \
  --mem-fraction-static 0.7 \
  --disable-piecewise-cuda-graph \
  --host 0.0.0.0 \
  --kv-cache-dtype bfloat16 \
  --moe-a2a-backend deepep --deepep-mode normal \
  --quantization fp8 --enable-dp-lm-head \
  --disable-radix-cache \
  --disable-shared-experts-fusion \
  --port 30001