#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Check if python3.10 and venv are installed
if ! command -v python3.10 &> /dev/null; then
    echo "python3.10 not found. Please install it before running this script."
    exit 1
fi

# Step 2: Install venv module if not already installed
if ! python3.10 -m venv --help &> /dev/null; then
    echo "Installing python3.10-venv module..."
    sudo apt update
    sudo apt install -y python3.10-venv
fi

# Step 3: Create virtual environment
ENV_NAME="ultralytics_env"
if [ -d "$ENV_NAME" ]; then
    echo "Virtual environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating virtual environment: $ENV_NAME"
    python3.10 -m venv $ENV_NAME
fi

# Step 4: Activate virtual environment
source $ENV_NAME/bin/activate
echo "Activated virtual environment: $ENV_NAME"

# Step 5: Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Step 6: Install PyTorch with CUDA 12.8 support
echo "Installing PyTorch (CUDA 12.8)..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Step 7: Install Ultralytics (without installing dependencies again)
echo "Installing Ultralytics (no extra dependencies)..."
pip install ultralytics --no-deps

# Install other core dependencies
pip install torch torchvision numpy matplotlib pandas pyyaml pillow psutil requests tqdm scipy seaborn ultralytics-thop

# Install headless OpenCV instead of the default
pip install opencv-python-headless



# Step 8: Confirm installation
echo ""
echo "Verifying installation..."

python - <<EOF
import torch
import torchvision
import torchaudio
import ultralytics

print("All modules imported successfully.")

print(f"torch version      : {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")
print(f"torchaudio version : {torchaudio.__version__}")
print(f"ultralytics version: {ultralytics.__version__}")

cuda_available = torch.cuda.is_available()
print(f"CUDA available     : {cuda_available}")

if cuda_available:
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU device         : {device_name}")
else:
    print("No GPU detected. Using CPU.")
EOF

echo "Environment setup complete!"


BOLD="\033[1m"
GREEN="\033[32m"
RESET="\033[0m"

echo -e "To activate the environment later, run:${BOLD}${GREEN} source ${ENV_NAME}/bin/activate"

