FROM vllm/vllm-openai:v0.9.1

RUN pip3 install flash_attn==2.8.0.post2
RUN pip3 install transformers==4.51.3

# Copy the model registration patch
COPY register_model.py /app/register_model.py

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
