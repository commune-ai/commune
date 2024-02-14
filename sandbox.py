import argparse
import commune as c

subspace = c.module('subspace')
config = subspace.config(to_munch=False)


urls ={'ws': [], 'http': []}
for i in range(1,5):
    name = f"commune-api-node-{i}.communeai.net"
    urls['ws'].append(f"wss://{name}/")
    urls['http'].append(f"https://{name}/")


config['urls']['commies'] = urls

subspace.save_config(config)

