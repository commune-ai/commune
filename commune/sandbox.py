import commune
# import langchain
# remote_module = commune.module('ReactAgentModule')
# print(remote_module.server_exists('ReactAgentModule'))
# import commune

class ModuleF:
    wrapped_module=True
    def bro(self):
        return 'bro'

agent = commune.module(ModuleF())
print(agent.get_module('langchain.agents.Tool'))
# print(commune.module('ReactAgentModule').run('What is the capital of the United States?'))
# agent.serve(wait_for_termination=True)
# print(commune.connect('ModuleF'))