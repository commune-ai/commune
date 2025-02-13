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