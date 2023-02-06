import commune
# import langchain
# remote_module = commune.module('ReactAgentModule')
# print(remote_module.server_exists('ReactAgentModule'))
# import commune


def test_create_actor():
    
    class DemoClass:
        fam = True

        def bro(self):
            return 'bro'
        
    print(DemoClass().__dict__)

    agent = commune.module(DemoClass)
    hasattr(agent, 'module_id')
    demo_module = commune.module(DemoClass)
    agent = commune.ray_launch(demo_module, refresh=True, virtual=True)
    module_id = agent.module_id
    assert commune.actor_exists(module_id) == True
    
    # call the function
    assert agent.bro() == 'bro'
    
    
    commune.kill_actor(agent.module_id)
    assert commune.actor_exists(module_id) == False
    
    

def test_wrapped_actor():
    
    class DemoClass:

        def bro(self):
            return 'bro'

    agent = commune.module(DemoClass)
    hasattr(agent, 'module_id')
    demo_module = commune.module(DemoClass)
    agent = commune.ray_launch(demo_module, refresh=True, virtual=True)
    module_id = agent.module_id
    assert commune.actor_exists(module_id) == True
    
    # call the function
    assert agent.bro() == 'bro'
    
    assert agent.wrapped_module == DemoClass.wrapped_module
    
    commune.kill_actor(agent.module_id)
    assert commune.actor_exists(module_id) == False
    
     
    

    
# print(commune.module('ReactAgentModule').run('What is the capital of the United States?'))
# agent.serve(wait_for_termination=True)
# print(commune.connect('ModuleF'))
test_create_actor()