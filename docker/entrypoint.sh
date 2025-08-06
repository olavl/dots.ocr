#!/bin/bash

echo "=== Starting DotsOCR vLLM server ==="

# Set environment variable to make vLLM aware of custom model
export PYTHONPATH=/workspace/weights/DotsOCR:$PYTHONPATH

# Run vLLM directly with trust-remote-code
exec python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/weights/DotsOCR \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --served-model-name dotsocr-model \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192
