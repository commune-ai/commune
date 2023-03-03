#/bin/bash

cd commune

# enter commune
# sudo make enter

# start the register loop
# pm2 start commune/block/bittensor/bittensor_module.py --name register_loop --interpreter python3 -- --fn register_loop

sudo docker exec -ti commune /bin/bash -c "python3 examples/register.py"
