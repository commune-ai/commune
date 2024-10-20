cli 


The commune cli needs to be able to call functions from the modules. This is a simple way to call functions from the modules.
c {modulename}/{fn} *args **kwargs

```bash
c serve module
```
To call the forward function of the model.openai module
```bash
c call module/ask hey
```

If you want to include positional arguments then do it 

```bash
c call module/ask hey stream=1
```

Limitatons

- Lists and dictionaries are not supported 
- Only positional arguments are supported
- Only one function can be called at a time