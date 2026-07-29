#!/bin/bash

set -e

##################################
# SGLang environment
##################################

cd /sgl-workspace/sglang || exit 1

export PYTHONPATH=$(pwd)/python:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=10,11


##################################
# Config
##################################

MODEL_PATH="/data/weights/CodeLlama-34b-hf"

LOG_DIR="/sgl-workspace/my_test/log"
RESULT_DIR="/sgl-workspace/my_test/result"

PORT=28887
rate=${rate:-16}

mkdir -p ${LOG_DIR}
mkdir -p ${RESULT_DIR}


SUMMARY=${RESULT_DIR}/summary.txt

echo "CodeLlama-34B Chunked Prefill Benchmark" > ${SUMMARY}
echo "======================================" >> ${SUMMARY}


# chunk size: 128 -> 8192
CHUNK_SIZES=(
128
256
512
1024
2048
4096
8192
)


##################################
# Experiment loop
##################################

for CHUNK_SIZE in ${CHUNK_SIZES[@]}
do

echo "======================================"
echo "Testing chunk size: ${CHUNK_SIZE}"
echo "======================================"


SERVER_LOG=${LOG_DIR}/server_chunk${CHUNK_SIZE}.log
RESULT_LOG=${RESULT_DIR}/bench_chunk${CHUNK_SIZE}.log


##################################
# Clean old process
##################################

echo "Cleaning old SGLang process..."

pkill -f "sglang serve.*--port $PORT" || true

sleep 10



##################################
# Start server
##################################

echo "Starting SGLang server..."

nohup numactl -C 80-159 \
sglang serve \
  --model-path ${MODEL_PATH} \
  --device npu \
  --attention-backend ascend \
  --tp-size 2 \
  --host 0.0.0.0 \
  --port ${PORT} \
  --mem-fraction-static 0.7 \
  --chunked-prefill-size ${CHUNK_SIZE} \
  --cuda-graph-bs 16 \
  > ${SERVER_LOG} 2>&1 &


SERVER_PID=$!

echo "Server PID=${SERVER_PID}"



##################################
# Wait model ready
##################################

echo "Waiting model ready..."

READY=0

for i in {1..120}
do

    RESPONSE=$(curl -s http://localhost:${PORT}/model_info || true)

    if [[ ${RESPONSE} == *"model_path"* ]]; then
        READY=1
        break
    fi

    sleep 5

done


if [ ${READY} -eq 0 ]; then

    echo "ERROR: Server not ready"
    echo "Check log:"
    echo ${SERVER_LOG}

    pkill -f "sglang serve" || true

    exit 1

fi


echo "Model ready!"



##################################
# Benchmark
##################################

echo "Start benchmark..."

curl http://localhost:28887/flush_cache

python benchmark/pdmux/bench_serving.py \
 --dataset-name loogle \
 --num-prompts 20 \
 --model /data/weights/CodeLlama-34b-hf \
 --backend sglang \
 --request-rate $rate \
 --port 28887 \
 > ${RESULT_LOG} 2>&1



echo "Benchmark finished"
echo "Result: ${RESULT_LOG}"



##################################
# Extract summary
##################################

echo "" >> ${SUMMARY}
echo "========== chunk=${CHUNK_SIZE} ==========" >> ${SUMMARY}

grep -E \
"Throughput|throughput|TPOT|TTFT|Output" \
${RESULT_LOG} >> ${SUMMARY} || true



##################################
# Stop server
##################################

echo "Stopping server..."

pkill -f "sglang serve" || true

sleep 20


echo "Finished chunk=${CHUNK_SIZE}"


done



echo "======================================"
echo "All experiments finished"
echo "Summary:"
echo ${SUMMARY}
echo "======================================"
