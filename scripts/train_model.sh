#!/bin/bash
source ./scripts/server_config.sh

export SSH_CONFIG="$(pwd)/$PROJECT_SSH_DIR/config"

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment>"
    exit 1
fi


EXPERIMENT_NAME=$1
SCREEN_NAME="training_$(date +%Y%m%d_%H%M%S)"

ssh -F $SSH_CONFIG project_server "cd $REMOTE_DIR && \
screen -dmS $SCREEN_NAME bash -c '\
source venv/bin/activate && \
python src/train.py --experiment $EXPERIMENT_NAME 2>&1 | tee training_log.txt'"

echo "Training started in screen session: $SCREEN_NAME"
echo "To check progress, use: ./scripts/check_training.sh $SCREEN_NAME"
