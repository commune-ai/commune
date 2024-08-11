import commune as c
import torch
import json
files = c.ls('./commune/module')
for f in files:
    filname = f.split('/')[-1]
    if filname.startswith('_'):
        new_name = 'module_' + filname[1:]