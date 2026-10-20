unset http_proxy
unset https_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

export PYTHONPATH=/home/l00993641/sglang/python:$PYTHONPATH

export HF_DATASETS_CACHE=/home/l00993641/eval_cache/hf_datasets
mkdir -p "$HF_DATASETS_CACHE"

evalscope eval \
    --model /home/weights/DeepSeek-V4-Flash-0731-w8a8 \
    --api-url http://127.0.0.1:30100/v1 \
    --api-key EMPTY \
    --eval-type openai_api \
    --generation-config '{
        "max_tokens": 84000,
        "seed": 3407,
        "top_p": 1.0,
        "temperature": 1.0,
        "n": 1,
        "timeout": 6000,
        "stream": true,
        "extra_body": {
            "chat_template_kwargs": {
                "thinking": true
            }
        }
    }' \
    --datasets gpqa_diamond  \
    --dataset-hub local \
    --dataset-args '{
        "gpqa_diamond": {
            "local_path": "/home/datasets/gpqa_diamond"
        }
    }' \
    --eval-batch-size 128 \
    --ignore-error


#  "reasoning_effort": "max"
#unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
#unset ALL_PROXY all_proxy
#
#export NO_PROXY=127.0.0.1,localhost
#export no_proxy=127.0.0.1,localhost
#
#MODEL_PATH=/home/weights/DeepSeek-V4-Flash-0731-w8a8
#
#evalscope eval \
#    --model ${MODEL_PATH} \
#    --api-url http://127.0.0.1:31000/v1 \
#    --api-key EMPTY \
#    --eval-type openai_api \
#    --generation-config '{"max_tokens":120000,"top_p":1,"temperature":1,"timeout":6000,"stream":true,"extra_body":{"chat_template_kwargs":{"thinking":true}}}' \
#    --datasets gpqa_diamond \
#    --dataset-args '{"gpqa_diamond":{"local_path":"/home/datasets/gpqa_diamond","subset_list":["gpqa_diamond"]}}' \
#    --eval-batch-size 32 \
#    --ignore-errors