
from tuwang import module

print(module.servers())

data = module.connect('dataset.bittensor')
sample = data(fn='sample')

sample['output_hidden_states'] = False
sample['topk'] = 10
sample['output_logits'] = False


model = module.connect('model.transformer::gptj:2')

print(model(fn='learn_step', kwargs = sample))