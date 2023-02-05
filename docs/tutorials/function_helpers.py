import commune


class ModuleF:
    wrapped_module=True
    def bro(self, fam:str, bro: dict):
        return 'bro'
    
    
agent = commune.module(ModuleF())

# you can also fetch the module from another file
# In this example you are fetching the ModuleF from the root directory of commune
# This root directory

ModuleF = commune.get_object('docs.tutorials.function_helpers.ModuleF')


# you can list the function
print('functions of ModuleF: ', agent.functions())

# you can get the function schema of a function
print('function schema of bro: ', agent.function_schema('bro'))

# you can get the function schema of a function
print('function schema map of bro: ', agent.function_schema_map())

