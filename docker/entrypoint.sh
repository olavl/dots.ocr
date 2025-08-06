#!/bin/bash

echo "=== Starting DotsOCR vLLM server ==="

# Patch the vllm CLI script to import DotsOCR model
echo "Patching vLLM to recognize DotsOCR model..."
sed -i '/^from vllm\.entrypoints\.cli\.main import main/a from DotsOCR import modeling_dots_ocr_vllm' $(which vllm)

echo "vllm script after patch:"
grep -A 1 'from vllm.entrypoints.cli.main import main' $(which vllm)

# Set Python path to include model directory
export PYTHONPATH=/workspace/weights/DotsOCR:$PYTHONPATH

echo "Starting vLLM server..."
exec vllm serve /workspace/weights/DotsOCR \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --served-model-name dotsocr-model \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192
