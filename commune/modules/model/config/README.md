
# Configuring Pipelines



## Config Loader

The config loader has some cool properties that you can use to compose your configurations 

local_copy(path) : 
- get a path with respect to root of config
copy(path)
- copy with respect to root of the global config (not your local root if another config is inheriting you)
get_cfg(path)
 - pull config giving its path with respect to the root of the commune
ENV! path/to/${ENV_VAR} 
- include env variables as well as local_var_dict = {} if put into the config.load function

