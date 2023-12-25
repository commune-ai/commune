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
import commune as c

module = c.get('module_tree')
c.print(module.get('subspace.chain'))
# import commune as c
# modules = {
#     'remote': c.connect('module'),
#     'local': c.module('module')()
# }
# info = {
#     'remote': 0,
#     'local': 0,
# }
# for module_key in modules.keys():
#     t0 = c.time()
#     r_remote = modules[module_key].info(schema=False)
#     t1 = c.time()
#     latency = t1 - t0
#     info[module_key] = latency

# c.print(info)
# # r_local = c.module('module').call(hardware=False)
# # c.print(r)



# r_local 