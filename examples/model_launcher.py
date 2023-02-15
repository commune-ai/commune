import commune
commune.new_event_loop()
# import streamlit as st

# servers = commune.servers()

# print(commune.launch('model.dendrite', tag='B' ))
import bittensor



# commune.servers()

# commune.get_module('commune.utils.math.MovingWindowAverage').test()
# print(commune.servers())
# print(commune.connect('BittensorDataset').sample())
# commune.launch('model.transformer', name='model', tag='gptj', fn='serve_module', device='2')
# commune.launch('model.transformer', name='model', tag='gptjt', fn='serve_module', device='0')
# commune.launch('model.transformer', name='model', tag='gpt2.7b', fn='serve_module', device='4')
# commune.launch('model.transformer',  name='gpt125m', tag='demo',  kwargs=dict(model_name='gpt125m'), mode='server')

# commune.get_module('model.transformer.gptneox')
# commune.launch('model.transformer.gptneox', name='gptneox', tag='0', kwargs=dict(model_name='gptneox'), mode='server')


# get the module map
print(commune.gpu_map())
# commune.launch('model.transformer.gptneox',  name='gpt20b', tag='0',  kwargs=dict(model_name='gpt20b'), mode='server')
# commune.launch('model.transformer', name='gptj', tag='trial_2', device='5', kwargs= {'tag': 'trial_2'}, mode='server')

# commune.launch('model.transformer', name='model', tag='gptj', fn='serve_module', device='2')



# commune.launch('model.transformer', name='train::gptj', fn='local_train', mode='pm2', 
#                device='3', kwargs={'model': 'gptj', 'tag': 'trial_2'} ,refresh=True)

# commune.launch('model.transformer', name='train::gptjt', fn='local_train', mode='pm2', 
#                device='5', kwargs={'model': 'gptjt', 'tag': 'trial_2'} ,refresh=True)


# commune.launch('model.transformer', name='train::gpt2.7b', fn='local_train', mode='pm2', 
#                device='4', kwargs={'model': 'gpt2.7b', 'tag': 'trial_2'} ,refresh=True)

# commune.launch('commune.model.transformer.gptneox.GPTNeoX', name='model::gpt20b', fn='serve_module', mode='pm2', 
#                device=None, kwargs={'model': 'gpt20b', 'max_memory' : {6: "70GiB" , 7: "20GiB"}} ,refresh=True)
