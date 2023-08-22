# import commune as c
# hf = c.module('hf')
# tokenizer_class = c.import_object('commune.model.transformer.llama.LlamaTokenizer')
# path = hf.get_model_snapshot('llama')
# tokenizer = tokenizer_class.from_pretrained(path)

# print(tokenizer.encode('hello world'))
import bittensor as bt
import commune as c
def thread_fleet(fn, n=10, tag=None,  *args, **kwargs):
    threads = []
    if tag == None:
        tag = ''
    for i in range(n):
        t = c.thread(fn=fn, tag=tag+i, *args, **kwargs)
    return c.thread_map


thread_fleet('fn')