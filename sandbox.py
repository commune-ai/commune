import commune as c


def register_servers(netuid=6, buffer = 100, timeout = 60):
    subspace = c.module('subspace')()
    stake = subspace.min_register_stake(netuid=netuid) + buffer
    key2balance = {k:v for k,v in c.key2balance().items() if v > stake}
    keys = list(key2balance.keys())
    c.print(f'Keys for registration: {keys}')
    servers = c.servers()
    c.print(f'Servers: {servers}')
    futures = []
    for i, s in enumerate(c.servers()):
        c.print(f'Registering {s}')
        key = keys[i % len(keys)]
        future = c.submit(c.register, kwargs=dict(name=s, netuid=netuid, stake=stake, key=key ), timeout=timeout)
        futures.append(future)

    for f in c.as_completed(futures, timeout=timeout):
        c.print(f.result())


while True:
    register_servers()
# for s in c.servers():
#     c.register(s, netuid=)