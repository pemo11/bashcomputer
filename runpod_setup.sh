#!/bin/bash

# RunPod Deployment Script for Bash Computer Agent
echo "🚀 Starting Bash Computer Agent setup on RunPod..."

# Check if we're running on GPU
if nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU detected! Setting up for local model deployment"
    export USE_LOCAL_MODEL=true
    export CUDA_VISIBLE_DEVICES=0
else
    echo "⚠️  No GPU detected. Running in API mode"
    export USE_LOCAL_MODEL=false
fi

# Install dependencies if not already installed
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Pre-download the model to avoid delays during first run
if [ "$USE_LOCAL_MODEL" = "true" ]; then
    echo "📥 Pre-downloading Llama model (this may take a few minutes)..."
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'meta-llama/Llama-3.1-8B-Instruct'
print(f'Downloading {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    device_map='auto'
)
print('✅ Model downloaded successfully!')
"
fi

echo "🎉 Setup complete! Starting Bash Computer Agent..."
echo "💡 Tip: The agent will ask for confirmation before executing any commands"
echo ""

# Start the agent with the RunPod-specific main file
python main_runpod.py