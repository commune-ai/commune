import commune as c
import torch

# meta = c.module('bittensor').get_metagraph(subtensor='local')



top_uid_map = c.print(c.module('bittensor').get_top_uids())

for uid, incentive in top_uid_map.items():
    print(uid, incentive)

