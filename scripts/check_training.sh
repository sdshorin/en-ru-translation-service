#!/bin/bash
source ./scripts/server_config.sh

export SSH_CONFIG="$(pwd)/$PROJECT_SSH_DIR/config"

SCREEN_NAME=$1

if [ -z "$SCREEN_NAME" ]; then
    echo "Active training sessions:"
    ssh -F $SSH_CONFIG project_server "screen -ls"
else
    echo "Latest training log:"
    ssh -F $SSH_CONFIG project_server "cd $REMOTE_DIR && tail -n 50 training_log.txt"
    
    PYTHON_PROCESS=$(ssh -F $SSH_CONFIG project_server "screen -S $SCREEN_NAME -Q select . > /dev/null 2>&1; echo \$?")
    
    if [ "$PYTHON_PROCESS" -eq "0" ]; then
        echo "Training is still running in session: $SCREEN_NAME"
    else
        echo "Training session completed or not found!"
    fi
fi
