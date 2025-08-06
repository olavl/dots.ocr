import sys
import os

# Add the model directory to path
model_path = "/workspace/weights/DotsOCR"
sys.path.insert(0, model_path)

# Import after vLLM is fully initialized
def register_dots_ocr():
    from vllm.model_executor.models import ModelRegistry

    # Change to the model directory to handle relative imports
    original_dir = os.getcwd()
    os.chdir(model_path)

    try:
        # Import the model class directly
        import modeling_dots_ocr_vllm
        DotsOCRForCausalLM = modeling_dots_ocr_vllm.DotsOCRForCausalLM

        # Register the model
        ModelRegistry.register_model("DotsOCRForCausalLM", DotsOCRForCausalLM)
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    import sys
    register_dots_ocr()

    # Now start vLLM with the model registered
    from vllm.entrypoints.openai.api_server import main
    sys.exit(main())
