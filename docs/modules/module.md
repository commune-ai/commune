A module is a colleciton of functions as well as data or state 
variables. The functions can be called from the command line interface or through the API. The data or state variables can be accessed and modified through the API. There is no blockchain or database in the module. The module is a simple way to organize code and data. To define a module, just define a class

To make a module create a name

{name}/module.py

Then do this

```python
class Module:
    def __init__(self):
        self.data = {}
    def forward(self, a=1, b=2):
        return a + b
```

To call it from the command line interface do this

```bash
c {name}/forward a=1 b=2
or 
c.module(name).forward(a=1, b=2)
```

