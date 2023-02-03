import commune
# import langchain
# remote_module = commune.module('ReactAgentModule')
# print(remote_module.server_exists('ReactAgentModule'))
# import commune
agent = commune.get_object('commune.block.langchain.agents.react.ReactAgentModule')
print(agent)
# from torch import Tensor