#!/bin/bash
source ./scripts/server_config.sh

export SSH_CONFIG="$(pwd)/$PROJECT_SSH_DIR/config"

SCREEN_NAME="training_$(date +%Y%m%d_%H%M%S)"

ssh -F $SSH_CONFIG project_server "cd $REMOTE_DIR && \
screen -dmS $SCREEN_NAME bash -c '\
source venv/bin/activate && \
python train.py $@ 2>&1 | tee training_log.txt'"

echo "Training started in screen session: $SCREEN_NAME"
echo "To check progress, use: ./scripts/check_training.sh $SCREEN_NAME"
