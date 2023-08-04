import commune as c

# please specify your own API key, and a unique tag that doesnt exist in the database
module = 'model.openai'
tag = 'unique_tag'

server_name = module + '::' + tag

assert server_name not in c.servers(network='subspace'), 'server already exists in commune'

# define the kwargs for initializing the module
kwargs = {'api_key': 'your_api_key'}


c.register(module=module, tag=tag, kwargs=kwargs)

