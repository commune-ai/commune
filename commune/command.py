

import commune

def run_miner(coldkey='fish',
              hotkey='1', 
              port=8101,
              subtensor = "194.163.191.101:9944",
              interpreter='python3',
              refresh: bool = True):
    
    name = f'miner_{coldkey}_{hotkey}'
    if refresh:
        commune.pm2_kill(name)
    command_str = f"pm2 start commune/model/client/model.py --name {name} --time --interpreter {interpreter} --  --logging.debug  --subtensor.chain_endpoint {subtensor} --wallet.name {coldkey} --wallet.hotkey {hotkey} --axon.port {port}"
    return commune.run_command(command_str)
print(run_miner())