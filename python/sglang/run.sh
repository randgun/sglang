export ASCEND_USE_FIA=1
export ASCEND_LAUNCH_BLOCKING=1
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
python3 -m sglang.launch_server --model-path /home/l00567497/weight/Qwen3-235B-A22B-Instruct-2507-W8A8/ --tp-size 8 --pcp-size 2 --trust-remote-code --attention-backend ascend --device npu --host 127.0.0.1 --port 5345 --mem-fraction-static 0.7  --chunked-prefill-size -1 --context-length 2048 --max-total-tokens 2048  --disable-radix-cache --max-running-requests 4   --disable-cuda-graph --dtype bfloat16