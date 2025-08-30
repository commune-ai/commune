# Development Container Service

A service that creates and manages isolated development containers with secure SSH access. Each container provides a preconfigured development environment accessible via SSH from your local machine.

## Features

- 🔒 Secure SSH key-based authentication
- 🐳 Isolated Docker containers
- 🛠️ Pre-installed development tools
- 🔑 Public key authentication only
- 📦 Easy container management

## System Requirements

### Remote Server
- Ubuntu/Debian Linux
- Python 3.9+
- Docker
- Available ports 15000-15100

### Local Machine
- Windows with Git Bash
- SSH client

## Setup Instructions

### 1. Local Machine Setup (Windows)

1. Open Git Bash and generate SSH key pair (if you don't have one):
```bash
ssh-keygen -t rsa -b 4096
```

2. Get your public key:
```bash
cat C:/Users/YourUsername/.ssh/id_rsa.pub
```

3. Copy the entire output - you'll need this for the server setup.

### 2. Remote Server Setup

1. Clone the repository and setup virtual environment:
```bash
git clone [https://github.com/BANADDA/cloudserver]
cd cloudserver
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
```bash
HOST=your.server.ip
API_PORT=8000
SSH_PORT_RANGE_START=15000
SSH_PORT_RANGE_END=15100
```

4. Update `test_api.py` with your public key:
```python
# Replace this with your public key from local machine
public_key = """ssh-rsa AAAA... YourUsername@ComputerName"""
```

## Usage

### Starting the Service

On the remote server:

1. Activate virtual environment:
```bash
source venv/bin/activate
```

2. Start the service:
```bash
uvicorn src.main:app --reload
```

### Creating and Accessing Containers

1. On the remote server, run:
```bash
python3 test_api.py
```

2. You'll see output like:
```
Container created successfully!
Container ID: abc123...
SSH Port: 15001
Username: devuser
Host: your.server.ip
```

3. From your local Windows machine, use the provided SSH command:
```bash
ssh -i C:/Users/YourUsername/.ssh/id_rsa -p 15001 devuser@your.server.ip
```

4. For connection debugging, use:
```bash
ssh -v -i C:/Users/YourUsername/.ssh/id_rsa -p 15001 devuser@your.server.ip
```

5. The container remains active until you press Enter in test_api.py

## Container Environment

Each container includes:
- Python 3.9 with development tools:
  - ipython
  - pylint
  - black
  - pytest
- Git
- Vim (configured for Python)
- Bash with useful aliases
- Dedicated workspace directory

## Project Structure
```
container_service/
├── docker/
│   └── dev_container/
│       ├── Dockerfile
│       ├── entrypoint.sh
│       └── configs/
│           ├── .bashrc
│           └── .vimrc
├── src/
│   ├── main.py
│   ├── models/
│   ├── services/
│   └── utils/
└── tests/
```

## API Endpoints

### Create Container
```http
POST /containers
Content-Type: application/json

{
    "public_key": "ssh-rsa AAAA..."
}
```

### Delete Container
```http
DELETE /containers/{container_id}
```

## Troubleshooting

### SSH Connection Issues

1. Verify key permissions on Windows:
```bash
chmod 600 C:/Users/YourUsername/.ssh/id_rsa
```

2. Check server-side container logs:
```bash
docker logs <container_id>
```

3. Use verbose SSH mode:
```bash
ssh -v -i C:/Users/YourUsername/.ssh/id_rsa -p <port> devuser@your.server.ip
```

### Common Problems

1. "Connection refused":
   - Verify container is running
   - Check port forwarding
   - Confirm server IP address

2. "Permission denied":
   - Check private key permissions
   - Verify public key in test_api.py
   - Ensure container was created successfully

3. "Host key verification failed":
   - Add the host to known_hosts or use:
   ```bash
   ssh-keyscan -p <port> <host> >> ~/.ssh/known_hosts
   ```

## Security Features

- No password authentication
- Root login disabled
- Container isolation
- Secure key permissions
- Non-root container user

## Development Commands

### Container Management
```bash
# List running containers
docker ps

# View container logs
docker logs <container_id>

# Stop container
docker stop <container_id>

# Remove container
docker rm <container_id>
```

### Service Management
```bash
# Start service
uvicorn src.main:app --reload

# Start service on specific port
uvicorn src.main:app --reload --port 8000

# Start service with host binding
uvicorn src.main:app --reload --host 0.0.0.0
```

## Best Practices

1. Always use SSH key authentication
2. Keep private keys secure
3. Use unique keys for different environments
4. Regularly update container images
5. Monitor container resources
6. Remove unused containers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request
