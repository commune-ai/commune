# Subspace Module

This involves a runthrough of the main function of subspace. We recommend going through the code if you want to understand the full extent of this module as there are alot of functions


## ROOT KEY

The root key is what you should be using for managing your commune modules. by default this is the following.

```bash 
c root_key
```
```bash
<Keypair (address=5GZBhMZZRMWCiqgqdDGZCGo16Kg5aUQUcpuUGWwSgHn9HbRC, path=module,  crypto_type: SR25519)>
```


## Register a Module
To register a module, you can use the following command

```
c model.openai register tag=sup api_key=sk-...
```

or

```
c register model.openai tag=sup api_key=sk-...
```

Please make sure you specify a unique tag, as it will not go through if someone else has that name on the subnet. When you deploy the module, the module will be serving locally on your machine and will be accessed by the network.


## Update a Module

To update a module, you can use the following command. At the moment you can update the module's name and address. Please not if you update the name, you will need to restart the server with the new name. This is currently something we want to avoid in the future by having to rename the server without killing it 

```

c update_module module=model.openai name=model.openai::fam1 address=124.545.545:8080 delegation_fee=10
```
