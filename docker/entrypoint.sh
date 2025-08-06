  #!/bin/bash
  set -e

  MODEL_PATH=${MODEL_PATH:-/workspace/weights/DotsOCR}
  export PYTHONPATH="$MODEL_PATH:$PYTHONPATH"

  echo "Patching vLLM to recognize DotsOCR model..."
  VLLM_INIT="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/__init__.py"

  # Add the import at the beginning of the file
  sed -i "1i import sys; sys.path.insert(0, '$MODEL_PATH'); from modeling_dots_ocr_vllm import DotsOCRForCausalLM" $VLLM_INIT

  echo "Starting vLLM server..."
  exec vllm serve "$MODEL_PATH" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.8 \
      --served-model-name dotsocr-model \
      --trust-remote-code \
      --host 0.0.0.0 \
      --port 8000
