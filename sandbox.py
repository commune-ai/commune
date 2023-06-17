import commune as c

uids = c.module('bittensor').query_map('Uids')[:10]
keys = {k.value:v.value for k,v in uids}
c.print(keys)