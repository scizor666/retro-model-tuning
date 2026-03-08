#!/bin/bash
set -e

echo "====================================="
echo "Retro Game FAQ Assistant - Setup"
echo "====================================="

echo "Requesting sudo privileges for system dependencies..."
# Update package list and install dependencies required to build Python with pyenv
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git

echo "Installing pyenv..."
if [ ! -d "$HOME/.pyenv" ]; then
    curl https://pyenv.run | bash
else
    echo "pyenv is already installed."
fi

echo "Downloading CUDA 12.1 toolkit installer..."
# Using the generic Linux installer for 12.1.1. It works across Ubuntu versions.
if [ ! -f "cuda_12.1.1_530.30.02_linux.run" ]; then
    wget -q -nc https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
fi

echo "Installing CUDA 12.1 Toolkit (silently, without overwriting existing 570 driver)..."
# Important: --toolkit flag ensures we DONT overwrite your 570 driver
sudo sh cuda_12.1.1_530.30.02_linux.run --silent --toolkit --override

echo "Adding CUDA and pyenv to profile (~/.zshrc and ~/.bashrc)..."
for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
    if [ -f "$rc" ]; then
        if ! grep -q 'pyenv' "$rc"; then
            echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$rc"
            echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> "$rc"
            echo 'eval "$(pyenv init -)"' >> "$rc"
        fi
        if ! grep -q 'cuda-12.1' "$rc"; then
            echo 'export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}' >> "$rc"
            echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> "$rc"
        fi
    fi
done

echo "System dependencies and toolkit installed successfully."
echo "Please re-source your profile or open a new terminal, run 'source ~/.zshrc'"
