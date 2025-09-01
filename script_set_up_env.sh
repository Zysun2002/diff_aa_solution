#!/bin/bash

# Configure tmux for better usability
cat > ~/.tmux.conf <<'EOF'
# Enable mouse support for scrolling and resizing
set -g mouse on

# Increase scrollback history limit
set-option -g history-limit 50000

# Use vi-style keys in copy mode (optional, feels like vim)
setw -g mode-keys vi
EOF

# Reload tmux config if already inside a tmux session
if [ -n "$TMUX" ]; then
    tmux source-file ~/.tmux.conf
    echo "✅ tmux config reloaded with mouse support and extended scrollback."
else
    echo "ℹ️ tmux config created. Restart or reload tmux to apply."
fi

set -e  # Exit immediately if a command fails

# Create a fresh conda environment (owned by you, not root)
ENV_NAME=diffvg
PYTHON_VERSION=3.10

echo "Creating conda environment '$ENV_NAME'..."
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

echo "Activating conda environment..."
# Important: use conda's shell hook to make 'conda activate' work inside scripts
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "Installing PyTorch and torchvision..."
conda install -y pytorch torchvision -c pytorch

echo "Installing numpy..."
conda install -y numpy

echo "Installing scikit-image..."
conda install -y scikit-image

echo "Installing cmake..."
conda install -y -c anaconda cmake

echo "Installing ffmpeg..."
conda install -y -c conda-forge ffmpeg

echo "Installing Python packages with pip..."
pip install -q svgwrite
pip install -q svgpathtools
pip install -q cssutils
pip install -q numba
pip install -q torch-tools
pip install -q visdom
pip install -q ipdb

echo "Running setup.py install..."
python setup.py install

echo "✅ All steps completed successfully in environment '$ENV_NAME'!"

echo "Reinstall ffmpeg..."
conda activate ENV_NAME
conda remove ffmpeg -y
conda install -c conda-forge 'ffmpeg>=4.3' -y
