import commune as c

# addresses = c.addresses('module', network='remote')
# c.print(c.call(addresses[0], fn='submit', kwargs={'fn': 'subspace.start_node', 'kwargs': {'node': 'alice'}}))
# c.print(c.connect('module').info())

s = c.module('subspace')()
vali_infos = s.vali_infos()

ps_map = c.module('remote').call('ps')
all_ps = []
empty_peers = [p for p, peers in ps_map.items() if len(peers) == 0]
c.print(empty_peers)
for ps in ps_map.values():
    all_ps.extend(ps)

vali_ps = sorted([p for p in all_ps if '.vali_' in p])

needed_valis = []
for vali_name, vali_info in vali_infos.items():
    if all([vali_name not in ps for ps in vali_ps]):
        needed_valis.append((vali_name, vali_info['ip']))



c.print(vali_ps)
c.print(needed_valis)
# c.print(vali_infos)
