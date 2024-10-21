<div align="center">

# **Commune AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Discord Chat](https://img.shields.io/badge/discord-join%20chat-blue.svg)](https://discord.com/invite/DgjvQXvhqf)
[![Website Uptime](https://img.shields.io/website-up-down-green-red/http/monip.org.svg)](https://www.communeai.org/)
[![Twitter Follow](https://img.shields.io/twitter/follow/communeaidotorg.svg?style=social&label=Follow)](https://twitter.com/communeaidotorg)

![Alt text](image.png)

FOR MORE INFO, GO TO [DOCS](./commune/docs) FOR MORE INFORMATION

Introduction to Commune

Commune is an open-source project that aims to create a network for connecting various developer tools. It's designed to be flexible and unopinionated, allowing developers to use it alongside their existing projects.

Key Features:
- Module Filesystem
- Subspace blockchain integration
- Flexible key management
- Pythonic CLI

A module is a class. The name of the module can be determined by the filepath with respect to the current working directory (c.pwd()/c pwd). 

make a class in a file


c new_module agi

Example (agi.py):
```python
class Agi(c.Module):
    def __init__(self, a=1, b=2):
        self.set_config(locals())

    def generate(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    
    forward = generate
```

You can call this module using:
Input
```bash 
c example/predict 10 # c {module}/{method} *args **kwargs
```
Result
```bash
⚡️⚡️⚡️⚡️predict⚡️⚡️⚡️⚡️
✅Result(0.001s)✅
11
```


# 1 Key Per Module

When you create a module, commune will see if the key name exists, and if it doesnt, it will generate a new one randomly. Dont worry, its not encrypted by default. 

Commune uses sr25519 keys for signing, encryption, and verification.

To add a new key:
```bash
c add_key alice 
```

```bash
c key alice
```

```bash
<Key(address=5CmA5vVC9s8uXE8htaigNicNokSouwLsSPPWMN3Z3V8uNEWw, path=alice, crypto_type=sr25519)>
```

To list keys:
```bash
c keys alice
```

```bash
[alice]
```


To sign a message:
```python
key = c.get_key("alice")
signature = key.sign("hello world")
```

## Serving

To serve a module:
```bash
c serve model.openai
```

To call a served module:
```python
c.call("model.openai/forward", "sup")
```


### Testing

To run tests:
```bash
pytest tests
```

Page 9: Contributing

Contributions to Commune are welcome. Please submit pull requests on the GitHub repository.

Page 10: License

Commune is licensed under MIT, but with a "Do What You Want" philosophy. The project encourages open-source usage without strict legal restrictions.

This documentation provides a high-level overview of Commune. For more detailed information on specific features, please refer to the individual module documentation or the project's GitHub repository.