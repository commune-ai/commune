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
    
    
def test_launch():
    # # print(commune.block.ray_actors())
    agentF = commune.module(ModuleF)
    agentG = commune.module(ModuleG)
    commune.launch(agentF, mode='ray')
    assert 'ModuleF' in commune.block.ray_actors()
    commune.launch(agentG, mode='ray')
    assert 'ModuleG' in commune.block.ray_actors()
    commune.kill_actor('ModuleG')
    assert 'ModuleG' not in commune.block.ray_actors()
    commune.kill_actor('ModuleF')
    
    
def test_gpu_allocation(gpus:int=1, cpus:int=1):
    agentF = commune.module(ModuleF)
    commune.launch(agentF, mode='ray', gpus=gpus, cpus=cpus)
    resources = commune.actor_resources('ModuleF')
    assert int(resources['gpus']) == gpus
    assert int(resources['cpus']) == cpus
    commune.kill_actor('ModuleF')
    # print(commune.get_actor('ModuleF'))

    # print(agent.connect('ModuleF').bro('fam'))

# # list actors
# print(commune.block.ray_actors())

# you can also fetch the module from another file
# In this example you are fetching the ModuleF from the root directory of commune
# This root directory

if __name__ == '__main__':
    test_gpu_allocation()
    # test_launch()
