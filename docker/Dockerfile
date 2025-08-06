  FROM vllm/vllm-openai:v0.9.1
  RUN pip3 install flash_attn==2.8.0.post2
  RUN pip3 install transformers==4.51.3

  # Add these lines:
  WORKDIR /app
  COPY entrypoint.sh /app/entrypoint.sh
  RUN chmod +x /app/entrypoint.sh

  ENV MODEL_PATH=/workspace/weights/DotsOCR
  ENV PYTHONPATH=/workspace/weights/DotsOCR:$PYTHONPATH

  EXPOSE 8000
  ENTRYPOINT ["/app/entrypoint.sh"]
