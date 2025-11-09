# Use NVIDIA's PyTorch image as base for GPU support
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables for RunPod deployment
ENV USE_LOCAL_MODEL=true
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH=/app

# Expose port (optional, for web interface if needed later)
EXPOSE 8000

# Default command - you can override this when launching the pod
CMD ["python", "main_langgraph.py"]