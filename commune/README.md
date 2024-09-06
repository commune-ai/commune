<div align="center">

# **Commune AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Discord Chat](https://img.shields.io/badge/discord-join%20chat-blue.svg)](https://discord.com/invite/DgjvQXvhqf)
[![Website Uptime](https://img.shields.io/website-up-down-green-red/http/monip.org.svg)](https://www.communeai.org/)
[![Twitter Follow](https://img.shields.io/twitter/follow/communeaidotorg.svg?style=social&label=Follow)](https://twitter.com/communeaidotorg)

![Alt text](image.png)
PLEASE REFER TO THE DOCS FOLDER FOR MORE INFO

[DOCS](./commune/docs)

Introduction to Commune

Commune is an open-source project that aims to create a network for connecting various developer tools. It's designed to be flexible and unopinionated, allowing developers to use it alongside their existing projects.

Key Features:
- Module Filesystem
- Subspace blockchain integration
- Flexible key management
- Pythonic CLI

To get started, you can install Commune either locally or using Docker.

Installation

Local Installation:
```bash
apt-get install python3.10 python3-pip npm
npm install -g pm2
pip install -r requirements.txt
pip install -e .
```

Docker Installation:
```bash
git clone https://github.com/commune-ai/commune.git
cd commune
make build
make start
make enter
```

After installation, sync with the network:
```bash
c ls
```

Page 3: Module Filesystem

Commune organizes modules in a filesystem-like structure. You can create local modules that integrate seamlessly with Commune's core modules.

Example:
```python
import commune as c

class Example(c.Module):
    def __init__(self):
        pass

    def predict(self, x):
        return x + 1
```

You can call this module using:
```bash
c model/predict x=1
```

Page 4: Subspace Integration

Commune uses the Subspace blockchain for:
- Decentralized Name Service (DNS) for deployed objects
- Stake-weighted voting system for performance evaluation

To register a module on the blockchain:
```bash
c register my_module_path name=my_module tag=1
```

Page 5: Key Management

Commune uses sr25519 keys for signing, encryption, and verification.

To add a new key:
```bash
c add_key alice
```

To list keys:
```bash
c keys
```

To sign a message:
```python
key = c.get_key("alice")
signature = key.sign("hello world")
```

Page 6: Pythonic CLI

Commune provides a Pythonic CLI for easy interaction:

```bash
c {module_name}/{function_name} *args **kwargs
```

Example:
```bash
c ls ./
```

This is equivalent to:
```python
import commune as c
c.ls('./')
```

Page 7: Serving Modules

To serve a module:
```bash
c serve model.openai::tag
```

To call a served module:
```python
c.call("model.openai::tag/forward", "sup")
```

Page 8: Testing

To run tests:
```bash
pytest commune/tests
```

Page 9: Contributing

Contributions to Commune are welcome. Please submit pull requests on the GitHub repository.

Page 10: License

Commune is licensed under MIT, but with a "Do What You Want" philosophy. The project encourages open-source usage without strict legal restrictions.

This documentation provides a high-level overview of Commune. For more detailed information on specific features, please refer to the individual module documentation or the project's GitHub repository.