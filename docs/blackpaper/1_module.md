
I am not the best at writing as I see ideas and its hard for me to put them down, unless i spend time. But here is a recap of commune. Our goal is to connect all applications into a decentralized network. This involves creating a tokenomics structure that can adapt to everything by providing incentives to run applicaitons. To do this, we required to 

The Module

A module can represent everything as it is turing complete. This means a module can represent any idea, like an ai model, to a business, to a student loan. In code terms, a module is a simple class in python, where it is a collection of functions that change a state. This general definition is the core foundation of the module.

The Module is Simple and Complex? 

Like a python class (to whic we try to maximally mimic as much as possible), a module can represent a simple idea like a function, or a complex idea like a business. This allows for a flexible network that can adapt to different use cases. The module is the core abstract that allows for a flexible network that can adapt to different use cases. This is becuase the module is turing complete, and can represent any idea.

Module Key

Each module is represented as a key which is an sr25519 key. The public key is used to register the module onto the blockchain. The key is used to sign, encrypt,decrypt and verify messages. These keys can also represent other keys on any other chain through transfering its seed entropy to another chain key format. This allows for modules to exist on any chain

![Alt text](image_module_key.png)

For Warning for Anti-Python Peeps

The module is designed from a python class, but this can be implemented in rust, javascript, or any other language, even if they are functional programming languages (via structs). You can think of the module as a class in any language, where it is a collection of functions that change a state. This is the core foundation of the module.

This is a simple example of a module that says whadup. 

```python
import commune as c
class Model(c.Module):
    def __init__(self, c=0):
        self.c = c
    
    def add(self, a=1, b=1):
        return a + b + self.c

```

![Alt text](image_module.png)


I can serve this as an api which runs in the background 

```python
c.serve(Whadup, name="mode_model")
```

calling whadup/ function and it will return a + b + c as defined inside the function. 

Serving a Module

Serving modules involves converting the python class into an http server. This server only exposes whitelist and blacklist functions, and hides powerful functions. You can adjust the whitelist and blacklist functions to your needs. 
When you serve the module, you will need to give it a name. By default it is 

```python
c.serve("model.openai::whadup")

```
This will serve the module onto the network. To register it onto the blockchain. The following stakes 100 tokens onto model.openai::whadup, onto the commune subnet.
```
c.register("model.openai::whadup", stake=100 ,subnet=commune)

Schema

The schema is the schema of the module's funcitons.

c wombo/schema

{
    '__init__': {
        'input': {'network': 'str'},
        'default': {'network': 'local'},
        'output': {},
        'docs': None,
        'type': 'self'
    },
    'testnet': {
        'input': {'miners': 'int', 'valis': 'int'},
        'default': {'miners': 3, 'valis': 1},
        'output': {},
        'docs': None,
        'type': 'self'
    }

c wombo/code


class Wombo(c.Module):
    def __init__(self, network = 'local'):
        self.set_config(locals()) # send locals() to init
    
    def testnet(self, miners=3, valis=1):
        miners =  
        valis = 
        results = []
        for s in miners + valis:
            results += [c.submit(c.serve, params=)]
        results = c.wait(results)
        return results


Shortcuts

If wombo is too long for you, ad you want w, set a shortcut.
This is stored in your root module's config.

c add_shortcut wombo w

{'success': True, 'msg': 'added shortcut (wombo -> w)'}


Namespaces

Namespaces are a way to organize modules. Each namespace is a collection of modules that associates. To see the namespace of a network

```python
c.namespace(network='local')
```
```bash
{'subnet.add.subnet.vali': '0.0.0.0:50214', 'subnet.vali::0': '0.0.0.0:50086', 'vali': '0.0.0.0:50204'}
```



