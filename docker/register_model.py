import sys
sys.path.insert(0, "/workspace/weights/DotsOCR")

# Import after vLLM is fully initialized
def register_dots_ocr():
    from vllm.model_executor.models import ModelRegistry
    from modeling_dots_ocr_vllm import DotsOCRForCausalLM

    # Register the model
    ModelRegistry.register_model("DotsOCRForCausalLM", DotsOCRForCausalLM)

if __name__ == "__main__":
    register_dots_ocr()

  # Now start vLLM with the model registered
    from vllm.entrypoints.openai.api_server import main
    main()
