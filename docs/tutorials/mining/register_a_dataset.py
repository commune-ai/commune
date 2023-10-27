import commune as c

# please specify your own API key, and a unique tag that doesnt exist in the database
module = 'data.hf'
tag = 'unique_tag'
dataset_name = 'truthful_qa'

kwargs = {'path': dataset_name}
 # default is data.hf.truthful_qa but we want to override this and avoid hf
name = f'data.{dataset_name}::{tag}'
server_name = module + '::' + tag
# define the kwargs for initializing the module

c.register(module=module, name=name, tag=tag, kwargs=kwargs)

