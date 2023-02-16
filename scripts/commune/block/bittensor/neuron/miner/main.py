from neuron import neuron

'''example

PYTHON

python commune/neuron/miner/main.py --logging.debug --subtensor.network nobunaga 
        --neuron.device cuda:0 --wallet.name {coldkey} --wallet.hotkey {hotkey} --logging.trace True
         --logging.record_log True --logging.logging_dir ~/.bittensor/miners --neuron.print_neuron_stats True

PM2


pm2 commune/neuron/miner/main.py --name miner_{coldkey}_{hotkey} -- --logging.debug --subtensor.network nobunaga 
        --neuron.device cuda:0 --wallet.name {coldkey} --wallet.hotkey {hotkey} --logging.trace True
         --logging.record_log True --logging.logging_dir ~/.bittensor/miners --neuron.print_neuron_stats True


'''


if __name__ == "__main__":
    template = neuron().run()
