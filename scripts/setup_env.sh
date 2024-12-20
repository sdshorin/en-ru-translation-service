#!/bin/bash
source ./scripts/server_config.sh

export SSH_CONFIG="$(pwd)/$PROJECT_SSH_DIR/config"

echo "Enter sudo password for remote server:"
read -s SUDO_PASSWORD

echo "Installing system dependencies..."
echo $SUDO_PASSWORD | ssh -F $SSH_CONFIG project_server "sudo -S apt update"

echo $SUDO_PASSWORD | ssh -F $SSH_CONFIG project_server "sudo -S apt install -y \
    python3.10-venv \
    python3-pip \
    build-essential \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6"

if [ $? -ne 0 ]; then
    echo "Error: Failed to install system dependencies!"
    exit 1
fi

echo "Setting up Python environment on remote server..."

ssh -F $SSH_CONFIG project_server "cd $REMOTE_DIR && \
    python3 -m venv venv && \
    source venv/bin/activate && \
    python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt"

if [ $? -eq 0 ]; then
    echo "Environment setup completed successfully!"
else
    echo "Error: Failed to setup environment!"
    ssh -F $SSH_CONFIG project_server "cd $REMOTE_DIR && cat venv/pip-log.txt"
    exit 1
fi
