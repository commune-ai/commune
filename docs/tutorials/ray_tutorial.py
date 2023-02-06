import commune


class ModuleF:
    def __init__(self,bro='bro'):
        self.bro = bro  
    def bro(self, fam:str):
        return 'bro'
    
class ModuleG:
    def __init__(self,bro='bro'):
        self.bro = bro  
    def bro(self, fam:str):
        return self.bro
    
# # print(commune.ray_actors())
agentF = commune.module(ModuleF)
agentG = commune.module(ModuleG)
print(agentF('bro').bro)
print(agentG('fam').bro)

# print(agent.connect('ModuleF').bro('fam'))

# # list actors
# print(commune.ray_actors())

# you can also fetch the module from another file
# In this example you are fetching the ModuleF from the root directory of commune
# This root directory

