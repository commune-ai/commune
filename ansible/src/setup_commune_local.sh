#/bin/bash


#/bin/bash
if [ -d "commune" ] 
then
    
    echo "Commune installed"

else
    git clone https://github.com/commune-ai/commune.git
fi

original_path=${PWD}

# ./scripts/install_npm_env.sh;

cd commune
sudo apt install python3-venv
python3 -m venv env;
# source env/bin/activate;
# ./scripts/install_python_env.sh;
# ./scripts/install_npm_env.sh;