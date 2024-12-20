#!/bin/bash
source ./scripts/server_config.sh

export SSH_CONFIG="$(pwd)/$PROJECT_SSH_DIR/config"
DATASET_DIR="./dataset"
ARCHIVE_NAME="dataset.tar.gz"

if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory '$DATASET_DIR' not found!"
    exit 1
fi

required_files=("trainval.csv" "trainval" "test")
for file in "${required_files[@]}"; do
    if [ ! -e "$DATASET_DIR/$file" ]; then
        echo "Error: Required file/directory '$file' not found in dataset!"
        exit 1
    fi
done

echo "Creating archive..."
tar czf "$ARCHIVE_NAME" \
    --use-compress-program="gzip -9" \
    --exclude=".*" \
    --exclude="__pycache__" \
    "$DATASET_DIR"

echo "Creating remote directory..."
ssh -F $SSH_CONFIG project_server "mkdir -p $REMOTE_DIR"

if ssh -F $SSH_CONFIG project_server "[ -d $REMOTE_DIR/dataset ]"; then
    echo "Dataset already exists on server. Checking if update needed..."
    
    LOCAL_SIZE=$(stat -f%z "$ARCHIVE_NAME")
    REMOTE_SIZE=$(ssh -F $SSH_CONFIG project_server "stat -f%z $REMOTE_DIR/$ARCHIVE_NAME 2>/dev/null || echo 0")
    
    if [ "$LOCAL_SIZE" = "$REMOTE_SIZE" ]; then
        echo "Dataset on server is up to date. Skipping upload."
        rm "$ARCHIVE_NAME"
        exit 0
    fi
fi

echo "Uploading dataset..."
rsync -avz --progress -e "ssh -F $SSH_CONFIG" \
    "$ARCHIVE_NAME" \
    project_server:$REMOTE_DIR/

echo "Extracting archive on remote server..."
ssh -F $SSH_CONFIG project_server "cd $REMOTE_DIR && tar xf $ARCHIVE_NAME && rm $ARCHIVE_NAME"

rm "$ARCHIVE_NAME"

echo "Dataset transfer completed!"