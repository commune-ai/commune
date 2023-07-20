import commune as c
import torch
import bittensor
from typing import List, Dict, Union, Tuple


cbt = c.module('bittensor')
reged = cbt.reged()
wallets = [cbt.get_wallet(r) for r in reged]
wallet = c.choice(wallets)


d = bittensor.text_prompting_pool(keypair=wallet.hotkey, metagraph=cbt.get_metagraph())
response = d.forward(roles=['system', 'assistant'], messages=['you are chat bot', 'what is the whether'], timeout=6)
c.print(response)