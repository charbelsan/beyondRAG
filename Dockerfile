# Example Dockerfile - Adapt to your actual base image and structure
# Use an appropriate base image with Python and potentially CUDA stub libraries
# e.g., FROM python:3.11-slim or FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
FROM python:3.11

# Set working directory
WORKDIR /app

# Install system dependencies if needed (e.g., build-essential, git)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     git \
#  && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY ./pipelines/requirements.txt /app/pipelines/requirements.txt

# <<< ADD PYTORCH INSTALLATION HERE >>>
# Install PyTorch, torchvision, torchaudio for CUDA 12.1
# Check https://pytorch.org/get-started/locally/ for the latest command for your specific CUDA/OS needs
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the requirements
# Use --no-cache-dir potentially to save space
RUN pip install --no-cache-dir -r /app/pipelines/requirements.txt

# Copy the rest of the application code
COPY . /app

# Set environment variables if needed (can also be done in docker-compose.yml)
# ENV NVIDIA_VISIBLE_DEVICES=all
# ENV PORT=9099

# Expose the port the application runs on (adjust if different)
EXPOSE 9080
# EXPOSE 9123 # If running standalone

# Command to run the application (adjust based on how Open WebUI runs it or if running standalone)
# This might be overridden by docker-compose.yml
# Example for standalone:
# CMD ["python", "-m", "pipelines.deepresearch_pipeline", "--host", "0.0.0.0", "--port", "9080"]
# Example for OpenWebUI (it might have its own entrypoint):
CMD ["echo", "Container ready. Entrypoint determined by Open WebUI or docker-compose."]

