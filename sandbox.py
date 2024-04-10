import commune as c

timeout = 40
futures = []
for s in c.stats(df=False):
    f = c.submit(c.add_profit_shares, kwargs=dict(key=s['name'], keys=['module']), timeout=timeout)
    futures.append(f)

for f in c.as_completed(futures, timeout=timeout):
    try:
        print(f.result())
    except Exception as e:
        print(e)

# root_key_address = c.root_key_address()
# key2future = {}
# netuids = c.netuids()
# timeout = 40
# for netuid in netuids:
#     my_stake_to = c.my_stake_to(netuid=netuid)
#     futures = []
#     for k, stake_to in my_stake_to.items():
#         if k!= root_key_address:
#             print('unstaking: ', k, stake_to, 'netuid: ', netuid)
#             futures += [c.submit(c.unstake_all, kwargs={'netuid': netuid, 'key': k })]

#     try:
#         for f in c.as_completed(futures, timeout=timeout):
#             try:
#                 print(f.result()) 
#             except Exception as e:
#                 print(e)
#     except Exception as e:
#         print(e)
        

