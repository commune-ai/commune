import commune


class ModuleF:
    wrapped_module=True
    def bro(self, fam:str, bro: dict):
        return 'bro'
    
commune.ray_init()

# print(commune.ray_actors())
# agent = commune.module(ModuleF(), return_class=True)

# list actors
print(commune.ray_actors())

# you can also fetch the module from another file
# In this example you are fetching the ModuleF from the root directory of commune
# This root directory

