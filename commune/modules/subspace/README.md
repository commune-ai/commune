# Commune Walkthrough

This is subspace, the blockchain of commune. This module allows you to start your own chain and interface with commune's chain. We will be going through the main functionalities.

Note, we will be refering to the apprviation of s as subspace.


## Register a module 
c register model.openai tag=sup api_key=YOURAPIKEY

This will convert the module into a server where model.openai::sup is the name. This will create a key with the name 
"model.openai::sup"

## Register a Validator

c register vali.text tag=whadup


Please make sure you have stake 

c s get_stake vali.text

If it is zero, please check a key without stake

To check the stake of all your keys

c s key2stake

or you can 







# syncing the chain

to sync the chain please do 

```bash
c sync 
or 
c s sync
```

This gets data from one chain

# to list your modules
c s my_modules


# 


