# import commune as c
# hf = c.module('hf')
# tokenizer_class = c.import_object('commune.model.transformer.llama.LlamaTokenizer')
# path = hf.get_model_snapshot('llama')c 
# tokenizer = tokenizer_class.from_pretrained(path)

# print(tokenizer.encode('hello world'))
import commune as c


module = c.connect('data.truthqa', network='local')
c.print(module.info())