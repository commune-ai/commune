import commune
import langchain
remote_module = commune.module('ReactAgentModule')
print(remote_module.server_exists('ReactAgentModule'))

print('BANDI')