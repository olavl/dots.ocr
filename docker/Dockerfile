FROM rednotehilab/dots.ocr:vllm-openai-v0.9.1

# Copy our entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
