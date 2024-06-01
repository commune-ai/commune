

## Deploying a Miner Tutorial

In this tutorial, you will learn how to deploy a validator on the network and perform various tasks related to staking, registration, and validation.

### Step 1: Registering a Miner

To register a validator with a specific tag, use the following CLI command:

```python 
import commune as c
model = c.serve('model.openai::whadup', api_key='.....') # default to local network
c.register('model.openai::whadup', subnet='commune') 
```
or 

INPUT
```bash
c serve model.openai::whadup # defaults to (netuid=0 subnet=commune key=model.openai::whadup)
c register vali::whadup # defaults to (netuid=0 subnet=commune key=module)
```

OUTPUT
Result
```bash
{'success': True, 'message': 'Successfully registered model.openai::whadup'}
```


Check the logs 
Now see if the miner is running

```bash
c logs model.openai::whadup

```


```bash
369|model. | INFO:     Started server process [51470]
369|model. | INFO:     Waiting for application startup.
369|model. | INFO:     Application startup complete.
369|model. | INFO:     Uvicorn running on http://0.0.0.0:50227 (Press CTRL+C to quit)
```

to call it 
INPUT

```bash
c model.openai::whadup forward sup
```
OR
```python
c.call("model.openai::whadup/forward", "sup")
```
OR
```python
model = c.connect('model.openai') # default to local network
model.forward('sup')
```

OUTPUT
```bash
Hey there! How can I assist you today?
```



if people are calling your module

```bash
c logs model.openai::whadup
```

```bash
1|model.openai  | INFO:     127.0.0.1:60231 - "POST /forward HTTP/1.1"
200 OK
1|model.openai  | INFO:     127.0.0.1:60572 - "POST /forward/ 
HTTP/1.1" 307 Temporary Redirect
1|model.openai  |  Success: model.openai::forward --> 
1|model.openai  | 5GZBhMZZRMWCiqgqdDGZCGo16Kg5aUQUcpuUGWwSgHn9HbRC... 
1|model.openai  | {
1|model.openai  |     'module': 'model.openai',
1|model.openai  |     'fn': 'forward',
1|model.openai  |     'address': 
'5GZBhMZZRMWCiqgqdDGZCGo16Kg5aUQUcpuUGWwSgHn9HbRC',
1|model.openai  |     'args': ['hey'],
1|model.openai  |     'kwargs': {}
1|model.openai  | }
1|model.openai  | INFO:     127.0.0.1:60572 - "POST /forward HTTP/1.1"
200 OK
```


to check servers

```python
c.servers()
```

['servers']

or 

```bash
c servers
```

to get the info


```python
c call model.openai/info
```


```bash
{
    'address': 'http://68.174.126.229:50113',
    'functions': [
        'blacklist',
        'is_admin',
        'generate',
        'whitelist',
        'schema',
        'info',
        'code',
        'server_name',
        'namespace',
        'fns'
    ],
    'attributes': [
        'config',
        'kwargs',
        'usage',
        'birth_time',
        'api_key',
        'prompt',
        'prompt_variables',
        '_key',
        'whitelist',
        'blacklist',
        'ip',
        'port',
        'address'
    ],
    'name': 'model.openai',
    'path': 'model.openai',
    'chash': 
'13919e7837259d158f47e18e2220fe79d7de4e57c06ee8534f458d5e6133eae7',
    'hash': 
'bafe034f13e19dd1b746d52f365b43836ff61d84aac90b7d612463a3d8506c9c',
    'signature': 
'52a8c7ea27ce5ec05cf55c0d086e6aa288004940c52dacdd5a92b944176646094b7d
dbbc6b1cf2d267366053ef26589f09dc730ef271ad3d18acc9ab4e83538a',
    'key': 
'5HarzAYD37Sp3vJs385CLvhDPN52Cb1Q352yxZnDZchznPaS',
    'schema': {
        'generate': {
            'input': {
                'prompt': 'str',
                'model': 'str',
                'presence_penalty': 'float',
                'frequency_penalty': 'float',
                'temperature': 'float',
                'max_tokens': 'int',
                'top_p': 'float',
                'choice_idx': 'int',
                'api_key': 'str',
                'role': 'str',
                'history': 'list',
                'stream': 'bool'
            },
            'default': {
                'prompt': 'sup?',
                'model': 'gpt-4-vision-preview',
                'presence_penalty': 0.0,
                'frequency_penalty': 0.0,
                'temperature': 0.9,
                'max_tokens': 4096,
                'top_p': 1,
                'choice_idx': 0,
                'api_key': None,
                'role': 'user',
                'history': None,
                'stream': False,
                'kwargs': None
            },
            'output': {},
            'docs': None,
            'type': 'self'
        }
    },
    'hardware': {
        'cpu': {'cpu_count': 12, 'cpu_type': 'arm'},
        'memory': {
            'total': 17.179869184,
            'available': 3.87670016,
            'used': 13.303169024,
            'free': 3.87670016,
            'active': 3.874586624,
            'inactive': 3.793354752,
            'percent': 77.4,
            'ratio': 0.774
        },
        'disk': {
            'total': 494.384795648,
            'used': 325.951094784,
            'free': 168.433700864
        },
        'gpu': {}
    },
    'namespace': {
        'model.openai': 'http://0.0.0.0:50113',
        'module': 'http://0.0.0.0:50198',
        'vali': 'http://0.0.0.0:50077'
    },
    'commit_hash': '9c3cdf050a29b973d2380675d59286504afad84b'
}
```


Updating Servers 


To update a module, use the following command:

This updates the module with the name 'model.openai::2' with a delegation fee of 10 and the address 'http://0.9:9888'.
```python
c.module('subspace')().update_module('model.openai', name='model.openai::2',delegation_fee=10, address='http://0.9:9888')
```



## Exta Stuff

# Updating All of your Servers at Once. 

Sometimes your server may be running on a different port than the one it is registered under. In order to update the servers, use the following command:

```python

c.module('subspace')().update_servers(search=None)
```

or 

c subspace/update_servers 


If you wnat to only update a specific server, you can use the following command:

```python
c.module('subspace')().update_servers(search='model.openai')
```
or 

c subspace/update_servers search=model.openai


### Launcher Keys

A launcher key is a key that is used to launch a miner. By default, when you register a module, the key used to stake and register the module is the root key (module). To make a launcher key, use the following command:

c new_key module::1

To list the keys

c launcher_keys 


To load the balance of the keys by an amount

c load_launcher_keys amount=500

This transfers the amount to the keys from the root key.




