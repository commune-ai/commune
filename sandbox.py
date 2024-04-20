import commune as c

servers = c.servers('subspace::')
timeout=100
subnet = 'subspace'
c.print(c.transfer_multiple(destinations=servers, amounts=50))

# futures = []
# for s in servers:
#     c.print(f"Registering {s} with key {s}")
#     future = c.submit(c.register, kwargs=dict(name=s, key=s, subnet=subnet),  timeout=timeout)
#     futures.append(future)

# for f in c.as_completed(futures, timeout=timeout):
#     print(f.result())


