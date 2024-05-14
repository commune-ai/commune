
The Module Tree: A filesystem for modules


The Tree class is designed to simplify the management of Python modules in a project with a potentially complex directory structure. By creating a centralized mapping between short, readable names and the actual file paths of the modules, it becomes much easier to navigate and use different parts of the codebase.

Here's a more detailed explanation of how the Tree class works:

**Parsing the Project Directory:**

**What is a tree?**
A tree is a directory structure that contains Python files. You can think of it as a folder where there is at least one Python file.

**What are the keys and values of a tree?**

keys: The keys of a tree are the simplified names of the Python files within the directory structure. These simplified names are generated using the path2simple method, which converts the file paths into a more readable format.
values: The values of a tree are the actual file paths of the Python files. These paths are used to import and access the modules within the project.

for example 

./model/openai.py -> model.openai
./model/transformer.py -> model.transformer
./model/transformer/encoder.py -> model.transformer.encoder


python```

import commune as c

class ModelOpenai:
    def __init__(self):
        print("ModelOpenAI initialized")
```

Can we have a module be a folder?

Yes, a module can be a folder. This can allow for modules to be further organized into submodules, making it easier to manage and navigate the codebase.

An example of a module folder structure:

model.openai -> model/openai/model_openai.py 

where 

python```

import commune as c

class ModelOpenai:
    def __init__(self):
        print("ModelOpenAI initialized")
```



What if the module is new and not in the tree?

Commune automatically finds if any path or folder follows the naming described above. If it does, then the object path can be inferred without explicitly adding it to the tree. If the module is in the tree, then the object path can be directly accessed using the tree by taking the key as an argument.


For instance if I am creating a new module

model/agi.py -> model.agi

python```

import commune as c

class ModelAgi(c.Module):
    def __init__(self):
        print("ModelAGI initialized")
    def forward(self, x):
        return x + 1
```

To call it

python```
model = c.module("model.agi")()
print(model.forward(1))
```

When you use a core module, you can directly access it using the tree.

python```
model = c.module("key")() # key is the key of the module in the tree
```


