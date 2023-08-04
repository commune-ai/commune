import commune as c
c.enable_jupyter()

s = c.module('subspace')()

# please specify your own API key, and a unique tag that doesnt exist in the database
module = 'model.openai'
tag = 'unique_tag'
server_name = module + '::' + tag

assert server_name not in s.servers(), 'server already exists in commune'

# define the kwargs for initializing the module

kwargs = {'api_key': 'your_api_key'}
c.register(module, tag=tag, kwargs=kwargs)

