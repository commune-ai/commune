# import commune as c
# hf = c.module('hf')
# tokenizer_class = c.import_object('commune.model.transformer.llama.LlamaTokenizer')
# path = hf.get_model_snapshot('llama')
# tokenizer = tokenizer_class.from_pretrained(path)

# print(tokenizer.encode('hello world'))
import bittensor as bt
import commune as c


module = c.connect('data.text.realfake', network='subspace')
c.print(module.sample())