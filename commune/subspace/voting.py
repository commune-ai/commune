import commune as c

Subspace = c.module('subspace')

class Voting(c.Module):

    #################
    #### Serving ####
    #################
    def add_subnet_proposal(
        self,
        netuid: int = None,
        key: str = None,
        network = 'main',
        nonce = None,
        **params,


    ) -> bool:

        self = Subspace()
            
        self.resolve_network(network)
        netuid = self.resolve_netuid(netuid)
        subnet_params = self.subnet_params( netuid=netuid , update=True, network=network)
        # remove the params that are the same as the module info
        params = {**subnet_params, **params}
        for k in ['name', 'vote_mode']:
            params[k] = params[k].encode('utf-8')
        params['netuid'] = netuid

        response = self.compose_call(fn='add_subnet_proposal',
                                     params=params, 
                                     key=key, 
                                     nonce=nonce)


        return response


