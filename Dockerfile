FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg espeak-ng libsndfile1 build-essential git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e ".[dev]"
ENV VCLAB_HOME=/workspace COQUI_TOS_AGREED=1
WORKDIR /workspace
EXPOSE 7860 8000
CMD ["vclab-webui", "--port", "7860"]
