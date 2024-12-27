#!/bin/bash
source ./scripts/server_config.sh

export SSH_CONFIG="$(pwd)/$PROJECT_SSH_DIR/config"

SCREEN_NAME=$1

if [ -z "$SCREEN_NAME" ]; then
    echo "Usage: $0 <screen_name>"
    echo "Available training sessions:"
    ssh -F $SSH_CONFIG project_server "screen -ls"
    exit 1
fi

SCREEN_LIST=$(ssh -F $SSH_CONFIG project_server "screen -ls | grep $SCREEN_NAME")

if [ -z "$SCREEN_LIST" ]; then
    echo "Error: Training session '$SCREEN_NAME' not found!"
    echo "Available training sessions:"
    ssh -F $SSH_CONFIG project_server "screen -ls"
    exit 1
fi

SESSION_COUNT=$(echo "$SCREEN_LIST" | wc -l)

if [ $SESSION_COUNT -gt 1 ]; then
    echo "Multiple sessions found:"
    echo "$SCREEN_LIST"
    echo ""
    echo "Please specify the full session ID. For example:"
    echo "$0 3851.training_20241227_133335"
    exit 1
fi

FULL_SESSION_NAME=$(echo "$SCREEN_LIST" | awk '{print $1}' | cut -d. -f1)

ssh -F $SSH_CONFIG project_server "screen -S $FULL_SESSION_NAME -X stuff $'\003'"
echo "Sent stop signal to training session: $FULL_SESSION_NAME"

sleep 5
SCREEN_STILL_EXISTS=$(ssh -F $SSH_CONFIG project_server "screen -ls | grep $FULL_SESSION_NAME")

if [ -n "$SCREEN_STILL_EXISTS" ]; then
    echo "Training session is still running. Do you want to force stop it? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
        ssh -F $SSH_CONFIG project_server "screen -S $FULL_SESSION_NAME -X quit"
        echo "Training session forcefully stopped"
    else
        echo "Training session left running"
    fi
else
    echo "Training session successfully stopped"
fi