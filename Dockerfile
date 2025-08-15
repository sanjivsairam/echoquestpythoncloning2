# Use the official Python slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/models
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

# Set workdir
WORKDIR /app

# Install system dependencies (MeCab and others)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    gcc \
    g++ \
    make \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app
COPY . /MeloTTS
COPY . /OpenVoice
COPY . /checkpoints_v2

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_clean.txt
RUN pip install ./MeloTTS
RUN python -c "import MeloTTS; print('Melotts loaded from', MeloTTS.__file__)"
RUN python -m unidic download


# Login to HuggingFace and download required models
RUN python -c "import os; from huggingface_hub import login; login(os.getenv('HUGGINGFACE_TOKEN'))" || true

# Download multiple Whisper models
RUN python -c "\
import os; from huggingface_hub import snapshot_download; \
snapshot_download('guillaumekln/faster-whisper-medium', token='<<update_with_proper_token>>'); \
snapshot_download('guillaumekln/faster-whisper-large-v2', token='<<update_with_proper_token>>')"

# Declare model cache volume
VOLUME ["/models"]


RUN pip install requests==2.27.1 huggingface_hub

# Expose port
EXPOSE 8080

# Start the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
