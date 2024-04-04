import commune as c
miners = 6
valis =  2
servers = [f'subnet.miner::{i}' for i in range(miners)]
servers += [f'subnet.vali::{i}' for i in range(valis)]
c.server_many(servers)