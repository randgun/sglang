curl http://192.168.0.207:30002/flush_cache

curl http://192.168.0.207:30002/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Write a Python function to calculate Fibonacci numbers.",
    "sampling_params": {
      "temperature": 0,
      "max_new_tokens": 8
    }
  }'
