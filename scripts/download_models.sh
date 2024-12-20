#!/bin/bash
source ./scripts/server_config.sh

export SSH_CONFIG="$(pwd)/$PROJECT_SSH_DIR/config"
LOCAL_DIR="./checkpoints"

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name>"
    echo "Available experiments:"
    ssh -F $SSH_CONFIG project_server "ls $REMOTE_DIR/checkpoints/"
    exit 1
fi

EXPERIMENT_NAME=$1
LOCAL_EXPERIMENT_DIR="$LOCAL_DIR/$EXPERIMENT_NAME"

if ! ssh -F $SSH_CONFIG project_server "[ -d $REMOTE_DIR/checkpoints/$EXPERIMENT_NAME ]"; then
    echo "Error: Experiment '$EXPERIMENT_NAME' not found on server"
    echo "Available experiments:"
    ssh -F $SSH_CONFIG project_server "ls $REMOTE_DIR/checkpoints/"
    exit 1
fi

mkdir -p $LOCAL_EXPERIMENT_DIR

rsync -avz -e "ssh -F $SSH_CONFIG" \
    project_server:$REMOTE_DIR/checkpoints/$EXPERIMENT_NAME/ \
    $LOCAL_EXPERIMENT_DIR/

echo "Models from experiment '$EXPERIMENT_NAME' downloaded to $LOCAL_EXPERIMENT_DIR"