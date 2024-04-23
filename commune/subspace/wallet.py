
import commune as c

class Wallet(c.m('subspace')):


    def __init__(self, network='main', **kwargs):
        self.set_config(locals())

    def call(self, *text, **kwargs):
        return self.talk(*text, **kwargs)


 
    def key_usage_path(self, key:str):
        key_ss58 = self.resolve_key_ss58(key)
        return f'key_usage/{key_ss58}'

    def key_used(self, key:str):
        return self.exists(self.key_usage_path(key))
    
    def use_key(self, key:str):
        return self.put(self.key_usage_path(key), c.time())
    
    def unuse_key(self, key:str):
        return self.rm(self.key_usage_path(key))
    
    def test_key_usage(self):
        key_path = 'test_key_usage'
        c.add_key(key_path)
        self.use_key(key_path)
        assert self.key_used(key_path)
        self.unuse_key(key_path)
        assert not self.key_used(key_path)
        c.rm_key('test_key_usage')
        assert not c.key_exists(key_path)
        return {'success': True, 'msg': f'Tested key usage for {key_path}'}
        


    #################
    #### Serving ####
    #################
    def update_global(
        self,
        key: str = None,
        network : str = 'main',
        sudo:  bool = True,
        **params,
    ) -> bool:

        key = self.resolve_key(key)
        network = self.resolve_network(network)
        global_params = self.global_params(fmt='nanos')
        global_params.update(params)
        params = global_params
        for k,v in params.items():
            if isinstance(v, str):
                params[k] = v.encode('utf-8')
        # this is a sudo call
        return self.compose_call(fn='update_global',
                                     params=params, 
                                     key=key, 
                                     sudo=sudo)




    #################
    #### Serving ####
    #################
    def vote_proposal(
        self,
        proposal_id: int = None,
        key: str = None,
        network = 'main',
        nonce = None,
        netuid = 0,
        **params,

    ) -> bool:

        self.resolve_network(network)
        # remove the params that are the same as the module info
        params = {
            'proposal_id': proposal_id,
            'netuid': netuid,
        }

        response = self.compose_call(fn='add_subnet_proposal',
                                     params=params, 
                                     key=key, 
                                     nonce=nonce)

        return response




    #################
    #### set_code ####
    #################
    def set_code(
        self,
        wasm_file_path = None,
        key: str = None,
        network = network,
    ) -> bool:

        if wasm_file_path == None:
            wasm_file_path = self.wasm_file_path()

        assert os.path.exists(wasm_file_path), f'Wasm file not found at {wasm_file_path}'

        self.resolve_network(network)
        key = self.resolve_key(key)

        # Replace with the path to your compiled WASM file       
        with open(wasm_file_path, 'rb') as file:
            wasm_binary = file.read()
            wasm_hex = wasm_binary.hex()

        code = '0x' + wasm_hex

        # Construct the extrinsic
        response = self.compose_call(
            module='System',
            fn='set_code',
            params={
                'code': code.encode('utf-8')
            },
            unchecked_weight=True,
            sudo = True,
            key=key
        )

        return response




    def unregistered_servers(self, search=None, netuid = 0, network = network,  timeout=42, key=None, max_age=None, update=False, transfer_multiple=True,**kwargs):
        netuid = self.resolve_netuid(netuid)
        network = self.resolve_network(network)
        servers = c.servers(search=search)
        key2address = c.key2address(update=1)
        keys = self.keys(netuid=netuid, max_age=max_age, update=update)
        uniregistered_keys = []
        unregister_servers = []
        for s in servers:
            if  key2address[s] not in keys:
                unregister_servers += [s]
        return unregister_servers



    def clean_keys(self, 
                   network='main', 
                   min_value=1,
                   update = True):
        """
        description:
            Removes keys with a value less than min_value
        params:
            network: str = 'main', # network to remove keys from
            min_value: int = 1, # min value of the key
            update: bool = True, # update the key2value cache
            max_age: int = 0 # max age of the key2value cache
        """
        key2value= self.key2value(netuid='all', update=update, network=network, fmt='j', min_value=0)
        address2key = c.address2key()
        rm_keys = []
        for k,v in key2value.items():
            if k in address2key and v < min_value:
                c.print(f'Removing key {k} with value {v}')
                c.rm_key(address2key[k])
                rm_keys += [k]
        return rm_keys
