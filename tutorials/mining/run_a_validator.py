import commune as c

vali = c.module('vali')
my_key = 'MY_KEY_PATH'
vali.serve(key=my_key, remote=True) # if you want to serve locally, set remote=False