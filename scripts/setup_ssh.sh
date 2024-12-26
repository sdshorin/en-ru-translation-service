#!/bin/bash
source ./scripts/server_config.sh

mkdir -p "$PROJECT_SSH_DIR"
chmod 700 "$PROJECT_SSH_DIR"

if [ -f "$PROJECT_SSH_DIR/$SSH_KEY_NAME" ]; then
    echo "SSH find"
else
    echo "create SSH '$SSH_KEY_NAME'..."
    ssh-keygen -t rsa -b 4096 -f "$PROJECT_SSH_DIR/$SSH_KEY_NAME" -N ""
fi

cat "$PROJECT_SSH_DIR/$SSH_KEY_NAME.pub" | ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" \
    "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"


cat > "$PROJECT_SSH_DIR/config" << EOL
Host project_server
    HostName $REMOTE_HOST
    User $REMOTE_USER
    Port $REMOTE_PORT
    IdentityFile $(pwd)/$PROJECT_SSH_DIR/$SSH_KEY_NAME
    StrictHostKeyChecking no
EOL

chmod 600 "$PROJECT_SSH_DIR/config"
echo "SSH setup completed!"
