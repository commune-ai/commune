## if not has npm install it
REPO_NAME=$(basename $(pwd))
OS_NAME=$(uname)
echo "REPO_NAME: $REPO_NAME OS_NAME: $OS_NAME"
# show its stats of the hard

if ! command -v npm &> /dev/null
then
    if [ "$OS_NAME" == "Linux" ]; then
        echo "Linux"
        sudo apt update
        sudo apt install npm
    fi
    if [ "$OS_NAME" == "Darwin" ]; then
        echo "Mac"
        brew install npm
    fi
    if [ "$OS_NAME" == "Windows" ]; then
        echo "Windows"
        choco install npm
    fi
fi

# make sure pm2 is installed
if ! command -v pm2 &> /dev/null
then
    npm install pm2 -g
fi
echo "PM2 installed"

# make sure python3 RUN apt-get install python3 python3-pip python3-venv -y

if ! command -v python3 &> /dev/null
then
    if [ "$OS_NAME" == "Linux" ]; then
        echo "Linux"
        sudo apt update
        sudo apt install python3 python3-pip python3-venv -y
    fi
    if [ "$OS_NAME" == "Darwin" ]; then
        echo "Mac"
        brew install python3
    fi
    if [ "$OS_NAME" == "Windows" ]; then
        echo "Windows"
        choco install python3
    fi
fi
echo "Python3 installed"

# ensure pip
if ! command -v pip3 &> /dev/null
then
    if [ "$OS_NAME" == "Linux" ]; then
        echo "Linux"
        sudo apt update
        sudo apt install python3-pip -y
    fi
    if [ "$OS_NAME" == "Darwin" ]; then
        echo "Mac"
        brew install python3-pip
    fi
    if [ "$OS_NAME" == "Windows" ]; then
        echo "Windows"
        choco install python3-pip
    fi
fi

echo "Pip3 installed"

# ensure the repo is installed as a python package by ckeckiing "pip list | grep $REPO_NAME" 

echo "Installed $REPO_NAME as a python package"


# ensure docker is installed

if ! command -v docker &> /dev/null
then
    echo "Docker not found. Installing Docker..."
    if [ "$OS_NAME" == "Linux" ]; then
        echo "Installing Docker on Linux"
        # Update package index
        sudo apt update
        # Install prerequisites
        sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
        # Add Docker's official GPG key
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
        # Add Docker repository
        sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
        # Update package index again
        sudo apt update
        # Install Docker
        sudo apt install -y docker-ce docker-ce-cli containerd.io
        # Add current user to docker group
        sudo usermod -aG docker $USER
        echo "Docker installed on Linux. Please log out and back in for group changes to take effect."
    fi
    if [ "$OS_NAME" == "Darwin" ]; then
        echo "Installing Docker on Mac"
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            echo "Homebrew not found. Please install Homebrew first."
            exit 1
        fi
        # Install Docker Desktop for Mac using Homebrew Cask
        brew install --cask docker
        echo "Docker Desktop installed on Mac. Please start Docker Desktop from Applications."
    fi
    if [ "$OS_NAME" == "Windows" ]; then
        echo "Installing Docker on Windows"
        # Install Docker Desktop for Windows using Chocolatey
        choco install docker-desktop -y
        echo "Docker Desktop installed on Windows. Please restart your computer and start Docker Desktop."
    fi
else
    echo "Docker is already installed"
fi

# ensure docker-compose is installed

if ! command -v docker-compose &> /dev/null
then
    echo "Docker Compose not found. Installing Docker Compose..."
    if [ "$OS_NAME" == "Linux" ]; then
        echo "Installing Docker Compose on Linux"
        # Download the latest stable release of Docker Compose
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        # Apply executable permissions
        sudo chmod +x /usr/local/bin/docker-compose
        # Create symbolic link (optional)
        sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
        echo "Docker Compose installed on Linux"
    fi
    if [ "$OS_NAME" == "Darwin" ]; then
        echo "Installing Docker Compose on Mac"
        # Docker Desktop for Mac includes Docker Compose, but if needed separately:
        brew install docker-compose
        echo "Docker Compose installed on Mac"
    fi
    if [ "$OS_NAME" == "Windows" ]; then
        echo "Installing Docker Compose on Windows"
        # Docker Desktop for Windows includes Docker Compose
        echo "Docker Compose is included with Docker Desktop on Windows"
    fi
else
    echo "Docker Compose is already installed"
fi

# Verify installations
echo ""
echo "=== Installation Summary ==="
if command -v docker &> /dev/null; then
    echo "Docker version: $(docker --version)"
else
    echo "Docker: NOT INSTALLED"
fi

if command -v docker-compose &> /dev/null; then
    echo "Docker Compose version: $(docker-compose --version)"
else
    echo "Docker Compose: NOT INSTALLED"
fi


# is commune installed

if pip3 list | grep commune
then
    IS_INSTALLED="True"
else
    IS_INSTALLED="False"
fi


# install commune
if [ "$IS_INSTALLED" == "False" ]; then
    echo "Installing commune"
    pip3 install -e .
else
    echo "commune is already installed"
fi

c key