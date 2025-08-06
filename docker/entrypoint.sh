#!/bin/bash

echo "=== Starting DotsOCR vLLM server ==="

#Set environment variable to make vLLM aware of custom model
export VLLM_CUSTOM_MODEL_PATH=/workspace/weights/DotsOCR

# Run the registration and server
exec python3 /app/register_model.py \
    --model /workspace/weights/DotsOCR \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --served-model-name dotsocr-model \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
