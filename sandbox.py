# import commune as c

# model = c.connect('model.hf.mistral7b_int4::3')
# t1 = c.time()
# vali = c.module('vali.text.truthqa')(start=False)

# text = 'what is the difference between metalica and school'
# c.print(vali.score_module(model))
# # output =  model.generate(text, max_new_tokens=max_new_tokens)
# # latency = c.time() - t1
# # tokens_per_second = max_new_tokens / latency
# # c.print(tokens_per_second, output)

from tqdm import tqdm
from time import sleep
import psutil

with tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm(total=100, desc='ram%', position=0) as rambar:
    while True:
        rambar.n=psutil.virtual_memory().percent
        cpubar.n=psutil.cpu_percent()
        rambar.refresh()
        cpubar.refresh()
        sleep(0.5)