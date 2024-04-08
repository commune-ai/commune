import commune as c



c.print(c.m)
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
        

