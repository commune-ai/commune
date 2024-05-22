import commune as c
key2address = c.key2address('openrouter') 
self_key_address = c.get_key('vali.text.realfake::tang')
my_keys = [k for k in c.m('s')().my_keys() if k != self_key_address]
c.vote()


