import sys
import os

# Add the model directory to path
model_path = "/workspace/weights/DotsOCR"
sys.path.insert(0, model_path)

def register_dots_ocr():
    from vllm.model_executor.models import ModelRegistry

    # Fix relative imports in the modeling file
    modeling_file = os.path.join(model_path, "modeling_dots_ocr_vllm.py")
    with open(modeling_file, 'r') as f:
        content = f.read()

    # Replace ALL relative imports with absolute imports
    content = content.replace('from .configuration_dots import', 'from configuration_dots import')
    content = content.replace('from .modeling_dots_vision import', 'from modeling_dots_vision import')
    content = content.replace('from .', 'from ')  # Catch any other relative imports

    # Write the fixed content to a temporary file
    fixed_file = os.path.join(model_path, "modeling_dots_ocr_vllm_fixed.py")
    with open(fixed_file, 'w') as f:
        f.write(content)

    # Import the fixed module
    import modeling_dots_ocr_vllm_fixed
    DotsOCRForCausalLM = modeling_dots_ocr_vllm_fixed.DotsOCRForCausalLM

    # Register the model
    ModelRegistry.register_model("DotsOCRForCausalLM", DotsOCRForCausalLM)

if __name__ == "__main__":
    register_dots_ocr()

    # Now start vLLM with the model registered
    from vllm.entrypoints.openai.api_server import main
    sys.exit(main())
