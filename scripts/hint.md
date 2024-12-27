
### Create server config

`server_config.sh`
```bash
#!/bin/bash

export REMOTE_USER="user"
export REMOTE_HOST=""
export REMOTE_PORT=""
export REMOTE_DIR="/home/$REMOTE_USER/image-classification"
export PROJECT_SSH_DIR="./.ssh"
export SSH_KEY_NAME="project_key"
```

### Run scripts

```sh
chmod +x *.sh
./scripts/setup_ssh.sh
./scripts/upload_project.sh
./scripts/setup_env.sh
# ./scripts/upload_dataset.sh
./scripts/train_model.sh  model=transformer training=transformer
./scripts/check_training.sh training_X_Y
./scripts/stop_training.sh training_X_Y
./scripts/download_models.sh transformer
```