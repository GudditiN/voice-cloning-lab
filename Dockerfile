FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg espeak-ng libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# CPU torch by default; the GPU override file passes a CUDA index instead.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir torch torchaudio --index-url ${TORCH_INDEX} \
    && pip install --no-cache-dir -e .

# Everything you create (and downloaded model weights) lives in /workspace, mounted from this folder.
ENV VCLAB_HOME=/workspace \
    COQUI_TOS_AGREED=1 \
    XDG_DATA_HOME=/workspace/models/.cache \
    HF_HOME=/workspace/models/.cache/hf
WORKDIR /workspace
EXPOSE 7860
CMD ["uvicorn", "vclab.server:app", "--host", "0.0.0.0", "--port", "7860"]
