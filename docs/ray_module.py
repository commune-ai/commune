# # import commune
# # import ray

import commune
commune.block.ray_init()
commune.get_actor('bruh')
# module = commune.create_actor(module=commune.get_module('dataset.text.huggingface'),name='bruh', refresh=True)
# print(commune.list_actors())
# # class Dummy:
# #     __name__ = 'fam'
# #     def __name2__(self):
# #         return self.__name__()
    
# # print(Dummy().__name__)