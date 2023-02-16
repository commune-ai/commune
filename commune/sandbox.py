import commune

# commune.log(commune.pm2_list(), 'green')
# print(commune.servers())
# print(commune.pm2_list())
# print(commune.servers())

# commune.launch('model.transformer.gptneox', name='model', tag='gpt20b', mode='server')

dataset = commune.connect('BittensorDataset')

for i in range(10):
    print(dataset.sample(no_tokenizer=True))
# model = commune.connect('model::gpt20b')

# print(model.forward(**dataset.sample(), no_grad= True,token_remap=True, output_length=10, topk=4096, output_logits=False, output_hidden_states=False))
# print(commune.launch('model.transformer', name='model', tag='gpt125', kwargs={'model_name': 'gpt125m'}))
# print(commune.module('commune.server.server.Server'))