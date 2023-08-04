import commune as c

# get the vali module
vali = c.module('vali')

# specify the keypath that repesents the vali module
my_key = 'MY_KEY_PATH'
vali.serve(key=my_key, remote=True) # if you want to serve locally, set remote=False