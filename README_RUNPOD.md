# RunPod Deployment Guide

This guide explains how to deploy your Bash Computer Agent on RunPod with local model support.

## 🚀 Quick Start on RunPod

1. **Launch a RunPod GPU Instance**:
   - Go to [runpod.ai](https://runpod.ai) and create an account
   - Choose a GPU instance (RTX 4090, A100, etc.)
   - Select PyTorch template or Ubuntu with CUDA
   - Launch the pod

2. **Upload Your Code**:
   ```bash
   # Clone or upload this repository to your RunPod instance
   git clone <your-repo> /workspace/bashcomputer
   cd /workspace/bashcomputer
   ```

3. **Run the Setup Script**:
   ```bash
   chmod +x runpod_setup.sh
   ./runpod_setup.sh
   ```

## 🔧 Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment for local model
export USE_LOCAL_MODEL=true
export CUDA_VISIBLE_DEVICES=0

# Run the agent
python main_runpod.py
```

## 🎯 Deployment Modes

### GPU Mode (Recommended for RunPod)
- Set `USE_LOCAL_MODEL=true`
- Uses local Llama-3.1-8B model on GPU
- No external API calls needed
- Faster inference, more privacy

### API Mode (Fallback)
- Set `USE_LOCAL_MODEL=false`
- Uses NVIDIA API (requires NVIDIA_API_KEY)
- Good for CPU-only instances

## 📋 Requirements

### For GPU Mode:
- GPU with at least 16GB VRAM (RTX 4090, A100, etc.)
- CUDA-enabled environment

### For API Mode:
- NVIDIA API key (set as `NVIDIA_API_KEY` environment variable)

## 🔑 Environment Variables

```bash
# For local model deployment
export USE_LOCAL_MODEL=true
export CUDA_VISIBLE_DEVICES=0

# For API fallback (optional)
export NVIDIA_API_KEY="your_nvidia_api_key"
```

## 🎮 Usage

Once running, the agent works exactly like your local version:
- Ask it to perform bash commands
- It will ask for confirmation before execution
- Type 'quit' to exit

Example:
```
['/' 🙂] list files in current directory
    ▶️   Execute 'ls -la'? [y/N]: y
```

## 🛠 Troubleshooting

### Model Loading Issues:
- Ensure you have enough GPU memory (16GB+ recommended)
- Try smaller model if needed (modify `local_model_name` in `config_runpod.py`)

### Dependency Issues:
- Run `pip install -r requirements.txt` again
- Check CUDA version compatibility

### API Mode Fallback:
- If GPU fails, set `USE_LOCAL_MODEL=false`
- Ensure NVIDIA API key is set

## 📁 File Structure

- `main_runpod.py` - RunPod-specific main file
- `config_runpod.py` - RunPod configuration
- `main_langgraph.py` - Your original local version (unchanged)
- `config.py` - Your original local config (unchanged)
- `runpod_setup.sh` - Automated setup script
- `Dockerfile` - For container deployment