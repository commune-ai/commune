## if not has npm install it
OS_NAME=$(uname)

echo "OS_NAME: $OS_NAME"

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

# make sure docker is installed

if ! command -v docker &> /dev/null
then
    if [ "$OS_NAME" == "Linux" ]; then
        echo "Linux"
        sudo apt update
        sudo apt install docker.io -y
        sudo systemctl start docker
        sudo systemctl enable docker
    fi
    if [ "$OS_NAME" == "Darwin" ]; then
        echo "Mac"
        brew install docker
    fi
    if [ "$OS_NAME" == "Windows" ]; then
        echo "Windows"
        choco install docker-desktop
    fi
fi
echo "Docker installed"