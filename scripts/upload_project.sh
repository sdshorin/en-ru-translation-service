#!/bin/bash
source ./scripts/server_config.sh

export SSH_CONFIG="$(pwd)/$PROJECT_SSH_DIR/config"

ssh -F $SSH_CONFIG project_server "mkdir -p $REMOTE_DIR"

rsync -avz \
--compress-level=9 \
--progress \
-e "ssh -F $SSH_CONFIG" \
--exclude 'outputs' \
--exclude '.env' \
--exclude 'venv' \
--exclude '__pycache__' \
--exclude '*.pyc' \
--exclude '.git' \
--exclude 'wandb' \
--exclude 'checkpoints' \
--exclude '.ssh' \
--delete \
./ project_server:$REMOTE_DIR/

echo "Project transfer completed!"
