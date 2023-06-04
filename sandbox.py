import commune as c
import torch

# meta = c.module('bittensor').get_metagraph(subtensor='local')


c.print(c.module('key')(seed='bro').__dict__)

for uid, incentive in top_uid_map.items():
    print(uid, incentive)

