import commune as c
import torch

# c.print(c.key_info('fam')['mnemonic'])

key2balance = c.key2balance()
c.print(key2balance)
key2balance.pop('module')

target_key = 'vali::fam'
target_key_address = c.key_info(target_key)['ss58_address']
buffer = 5
futures = []
timeout=100
for key, balance in key2balance.items():
    amount = int(balance - buffer)
    if amount <= 0 or key == target_key:
        continue
    
    params  = {'dest': target_key_address, 'amount': amount,  'key': key , 'timeout': timeout}
    print(params)
    futures += [c.submit(c.transfer,params)]

for f in c.as_completed(futures, timeout=timeout):
    c.print(f.result())

# target_key = 