import commune as c
import torch

# meta = c.module('bittensor').get_metagraph(subtensor='local')
c.call('model.diffusion', 'predict_txt2img', prompt='a photo of an astronaut riding a horse on mars', device='cuda')
