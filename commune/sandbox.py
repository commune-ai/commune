import commune

# commune.log(commune.pm2_list(), 'green')
# print(commune.servers())
# print(commune.pm2_list())
# print(commune.servers())

for m in ['gpt2.7b', 'gptjt', 'gpt125m']:
    commune.launch('model.transformer', name='model', tag=m, kwargs={'model_name': m}, mode='server')

# print(commune.launch('model.transformer', name='model', tag='gpt125', kwargs={'model_name': 'gpt125m'}))
# print(commune.module('commune.server.server.Server'))