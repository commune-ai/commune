
Commune is a modular framework

Here are your modules 

c modules
    
'''
[
    'model.openai',
    'model.gpt3',
    'model.gpt2',
    ...
]
```


Once you find a module you want to use, you can play with it in the terminal or use it in your code.

To play with it in the terminal, you can use the following commands

```python

# call the forward function of the module with the input "whadup" (args, and kwargs are supported)
c model.openai forward "whadup" 
# or 
c model.openai forward text=whadup

```


If you are unsure of what the inputs are, you can use the following commands

To get the function signatures

```bash

c model.openai schema forward defaults=True

```




```python
c.module('model.openai').schema(*args, **kwargs)
```

OUTPUT
```bash
{
    'forward': {
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
            'retry': 'bool',
            'role': 'str',
            'history': 'list'
        },
        'default': {
            'self': None,
            'prompt': 'sup?',
            'model': 'gpt-3.5-turbo',
            'presence_penalty': 0.0,
            'frequency_penalty': 0.0,
            'temperature': 0.9,
            'max_tokens': 100,
            'top_p': 1,
            'choice_idx': 0,
            'api_key': None,
            'retry': True,
            'role': 'user',
            'history': None,
            'kwargs': None
        },
        'output': 'str',
        'type': 'self'
    }
}

```



To serve a module, you can use the following command

```bash
c model.openai serve tag=sup
```

```python
c.module('model.openai').serve(tag='sup', api_key='sk....', **kwargs)
# or 
c.serve('model.openai', tag='sup', api_key='sk....', **kwargs )
```


To connect to a module, you can use the following command


Way 1 
```python

module = c.connect('model.openai')
# call the forward function of the module with the input "whadup" (args, and kwargs are supported)
output = module.forward("whadup")
c.print(output) # custom print that wraps Console Log for pretty printgs


```

Way 2
```bash
output = c.call('model.openai', 'forward', text='whadup')

```
To get the info of the server

```bash

c model.openai info
```



```python


When this happens you are overwriting the the default values of the modules that is specified in the config

c model.openai config

To change the config, you can go to the file

 
```
c model.openai configpath
```

```
~/commune/commune/modules/model/openai/model_openai.yaml
```




