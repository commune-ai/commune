#/bin/bash
# run ray node
ray start --head --port=${RAY_PORT};
# run the api
nohup python commune/gradio/api/module.py --api &> api.out&

# start jupyter notebook
nohup jupyter notebook --ip 0.0.0.0 --no-browser --allow-root&> jupyter.out&
# pip install https://github.com/opentensor/cubit/releases/download/v1.1.1/cubit-1.1.1-cp38-cp38-linux_x86_64.whl;
tail -F anything;

