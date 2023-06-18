import commune as c
subspace = c.module('subspace')
netuid = 1
uids = subspace.query_map('Uids')
namespace = subspace.query_map('Namespace')
address = subspace.query_map('Address')

keys = {k.value:v.value for k,v in uids}
c.print(keys)