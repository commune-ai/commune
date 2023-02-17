import commune

# commune.log(commune.pm2_list(), 'green')
# print(commune.servers())
# print(commune.pm2_list())
# print(commune.servers())

# import commune
# commune.new_event_loop()
# import streamlit as st

# servers = commune.servers()

commune.launch('dataset.text.bittensor', name='dataset.bittensor', mode='server')
# print(commune.launch('model.dendrite', name='model::dendrite', mode='ray'))
# import bittensor

# for model in [ 'gpt2.7b', 'gpt125m', 'gptj', 'gptjt','opt13b']:
#     # commune.pm2_kill('model::'+model)
#     commune.launch('model.transformer', name='model', tag=model, fn='serve_module', kwargs={'model_name': model})

# for model in [ 'gpt20b']:
#     # commune.pm2_kill('model::'+model)
#     commune.launch('model.transformer.gptneox', name='model', tag=model, fn='serve_module', kwargs={'model_name': model})

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

# print(commune.server_exists('module'))
# print(commune.connect('module').pm2_list())
# print(commune.connect('module').launch('block.bittensor',fn='register_loop', mode='pm2'))
# dataset = commune.connect('BittensorDataset')

# for i in range(10):
#     print(dataset.sample(no_tokenizer=True))
# model = commune.connect('model::gpt20b')

# print(model.forward(**dataset.sample(), no_grad= True,token_remap=True, output_length=10, topk=4096, output_logits=False, output_hidden_states=False))
# print(commune.launch('model.transformer', name='model', tag='gpt125', kwargs={'model_name': 'gpt125m'}))
# print(commune.module('commune.server.server.Server'))