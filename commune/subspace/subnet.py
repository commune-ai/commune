from typing import *
import commune as c
import requests
class SubspaceSubnet:
    
    subnet_param_features = [
        "ImmunityPeriod",
        "MinAllowedWeights",
        "MaxAllowedWeights",
        "Tempo",
        "MaxAllowedUids",
        "TargetRegistrationsInterval",
        "TargetRegistrationsPerInterval",
        "MaxRegistrationsPerInterval",
        "Founder",
        "FounderShare",
        "IncentiveRatio",
        "TrustRatio",
        "SubnetNames",
        "MaxWeightAge",
        "BondsMovingAverage",
        "MaximumSetWeightCallsPerEpoch",
        "AdjustmentAlpha",
        "MinImmunityStake",
    ]



    def stake_to(self, netuid = 0,block=None,  max_age=1000, update=False, fmt='nano',**kwargs):
        stake_to = self.query_map('StakeTo', block=block, max_age=max_age, update=update,  **kwargs)
        format_value = lambda v:  {v_k: self.format_amount(v_v, fmt=fmt) for v_k, v_v in v.items()}
        stake_to = {k: format_value(v) for k,v in stake_to.items()}
        return stake_to
    

    def stake_from(self, 
                    block=None, 
                    update=False,
                    max_age=10000,
                    fmt='nano', 
                    **kwargs) -> List[Dict[str, Union[str, int]]]:
        
        stake_from = self.query_map('StakeFrom', block=block, update=update, max_age=max_age )
        format_value = lambda v:  {v_k: self.format_amount(v_v, fmt=fmt) for v_k, v_v in v.items()}
        stake_from = {k: format_value(v) for k,v in stake_from.items()}
        return stake_from

    """ Returns network Tempo hyper parameter """
    def stakes(self, fmt:str='j', max_age = 100, update=False, **kwargs) -> int:
        stake_from =  self.stake_from( update=update, max_age=max_age, fmt=fmt,)
        stakes = {k: sum(v.values()) for k,v in stake_from.items()}
        return stakes
    
    def leaderboard(self, netuid = 0, block=None, update=False, columns = ['emission', 'name', 'incentive', 'dividends'], **kwargs):
        modules = self.get_modules(netuid=netuid, block=block, update=update, **kwargs)
        return c.df(modules)[columns]

    
    def min_stake(self, netuid: int = 0, fmt:str='j', **kwargs) -> int:
        min_stake = self.query('MinStake', netuid=netuid,  **kwargs)
        return self.format_amount(min_stake, fmt=fmt)
    
    def regblock(self, netuid: int = 0, block: Optional[int] = None,  update=False ) -> Optional[float]:
        regblock =  self.query_map('RegistrationBlock',block=block, update=update )
        if isinstance(netuid, int):
            regblock = regblock[netuid]
        return regblock

    def emissions(self, netuid = 0, block=None, update=False, fmt = 'nanos', **kwargs):

        emissions = self.query_vector('Emission',  netuid=netuid, block=block, update=update, **kwargs)
        if netuid == 'all':
            for netuid, netuid_emissions in emissions.items():
                emissions[netuid] = [self.format_amount(e, fmt=fmt) for e in netuid_emissions]
        else:
            emissions = [self.format_amount(e, fmt=fmt) for e in emissions]
        
        return emissions

    emission = emissions


    def total_emission( self, netuid: int = 0, block: Optional[int] = None, fmt:str = 'j', **kwargs ) -> Optional[float]:
        total_emission =  sum(self.emission(netuid=netuid, block=block, **kwargs))
        return self.format_amount(total_emission, fmt=fmt)

    def addresses(self, netuid: int = 0, update=False, **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
        addresses = self.query_map('Address',netuid=netuid, update=update, **kwargs)
        
        if isinstance(netuid, int):
            addresses = list(addresses.values())
        else:
            for k,v in addresses.items():
                addresses[k] = list(v.values())
        return addresses

    def namespace(self, search=None, netuid: int = 0, update:bool = False, timeout=30, local=False, max_age=1000, **kwargs) -> Dict[str, str]:
        namespace = {}  
        results = {
            'names': None,
            'addresses': None
        }
        netuid = self.resolve_netuid(netuid)
        while any([v == None for v in results.values()]):
            future2key = {}
            for k,v in results.items():
                if v == None:
                    f =  c.submit(getattr(self, k), kwargs=dict(netuid=netuid, update=update, max_age=max_age, **kwargs))
                    future2key[f] = k
            for future in c.as_completed(list(future2key.keys()), timeout=timeout):
                key = future2key.pop(future)
                r = future.result()
                if not c.is_error(r) and r != None:
                    results[key] = r
        
        if netuid == 'all':
            netuid2subnet = self.netuid2subnet()
            namespace = {}
            for netuid, netuid_addresses in results['addresses'].items():
                for uid,address in enumerate(netuid_addresses):
                    name = results['names'][netuid][uid]
                    subnet = netuid2subnet[netuid]
                    namespace[f'{subnet}/{name}'] = address

        else:
            namespace = {k:v for k,v in zip(results['names'], results['addresses'])}

        if search != None:
            namespace = {k:v for k,v in namespace.items() if search in str(k)}

        if local:
            ip = c.ip()
            namespace = {k:v for k,v in namespace.items() if ip in str(v)}

        return namespace

    def emissions(self, netuid = 0, network = "main", block=None, update=False, **kwargs):
        return self.query_vector('Emission', network=network, netuid=netuid, block=block, update=update, **kwargs)

    def key2name(self, key: str = None, netuid: int = 0) -> str:
        modules = self.keys(netuid=netuid)
        key2name =  { m['key']: m['name']for m in modules}
        if key != None:
            return key2name[key]

    def uid2name(self, netuid: int = 0, update=False,  **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
        names = self.query_map('Name', netuid=netuid, update=update,**kwargs)
        return names

    def keys(self,
             netuid = 0,
              update=False, 
              max_age=1000,
             **kwargs) -> List[str]:
        keys =  self.query_map('Keys', netuid=netuid, update=update,  max_age=max_age, **kwargs)
        if netuid == 'all':
            for netuid, netuid_keys in keys.items():
                keys[netuid] = list(netuid_keys.values())
        else:
            keys = list(keys.values())
        return keys

    def delegation_fee(self, netuid = 0, block=None, update=False, fmt='j'):
        delegation_fee = self.query_map('DelegationFee', netuid=netuid, block=block ,update=update)
        return delegation_fee


    def subnet_params(self, 
                    netuid=0,
                    update = False,
                    max_age = 1000,
                    timeout=40,
                    fmt:str='j', 
                    features  = None,
                    value_features = [],
                    **kwargs
                    ) -> list:  
        
        features = features or self.subnet_param_features
        netuid = self.resolve_netuid(netuid)
        path = f'query/{self.network}/SubspaceModule.SubnetParams.{netuid}'          
        subnet_params = self.get(path, None, max_age=max_age, update=update)
        names = [self.feature2name(f) for f in features]
        future2name = {}
        name2feature = dict(zip(names, features))
        for name, feature in name2feature.items():
            if netuid == 'all':
                query_kwargs = dict(name=feature, block=None, max_age=max_age, update=update)
                fn = self.query_map
            else:
                query_kwargs = dict(name=feature, 
                                    netuid=netuid,
                                     block=None, 
                                     max_age=max_age, 
                                     update=update)
                fn = self.query
            f = c.submit(fn, kwargs=query_kwargs, timeout=timeout)
            future2name[f] = name
        
        subnet_params = {}

        for f in c.as_completed(future2name, timeout=timeout):
            result = f.result()
            subnet_params[future2name.pop(f)] = result
        for k in value_features:
            subnet_params[k] = self.format_amount(subnet_params[k], fmt=fmt)

        if netuid == 'all':
            subnet_params_keys = list(subnet_params.keys())
            for k in subnet_params_keys:
                netuid2value = subnet_params.pop(k)
                for netuid, value in netuid2value.items():
                    if netuid not in subnet_params:
                        subnet_params[netuid] = {}
                    subnet_params[netuid][k] = value
        return subnet_params

    def pending_deregistrations(self, netuid = 0, update=False, **kwargs):
        pending_deregistrations = self.query_map('PendingDeregisterUids',update=update,**kwargs)[netuid]
        return pending_deregistrations
    
    def num_pending_deregistrations(self, netuid = 0, **kwargs):
        pending_deregistrations = self.pending_deregistrations(netuid=netuid, **kwargs)
        return len(pending_deregistrations)
        
    def subnet_names(self , search=None, update=False, block=None, max_age=60, **kwargs) -> Dict[str, str]:
        records = self.query_map('SubnetNames', update=update,  block=block, max_age=max_age, **kwargs)
        subnet_names = sorted(list(map(lambda x: str(x), records.values())))
        if search != None:
            subnet_names = [s for s in subnet_names if search in s]
        return subnet_names
    

    def subnets(self, **kwargs) -> Dict[int, str]:
        return self.subnet_names(**kwargs)
    
    def num_subnets(self, **kwargs) -> int:
        return len(self.subnets(**kwargs))
    
    def subnet2stake(self, fmt='j'):
        netuid2subnet = self.netuid2subnet()
        netuid2stake = self.netuid2stake(fmt=fmt)
        subnet2stake = {}
        for netuid, subnet in netuid2subnet.items():
            subnet2stake[subnet] = netuid2stake[netuid]
        return subnet2stake
        

    def netuid2stake(self, fmt='j',  **kwargs):
        netuid2stake = self.query_map('TotalStake',  **kwargs)
        for netuid, stake in netuid2stake.items():
            netuid2stake[netuid] = self.format_amount(stake, fmt=fmt)
        return netuid2stake
    
    def netuid2n(self, fmt='j',  **kwargs):
        netuid2n = self.query_map('N',  **kwargs)
        return netuid2n

    def trust(self, 
                  netuid = 0, 
                  block=None,  
                  update:bool = False, 
                  **kwargs):
        return self.query_vector('Trust', netuid=netuid,  block=block, update=update, **kwargs)

    def incentives(self, 
                  netuid = 0, 
                  block=None,  
                  update:bool = False, 
                  **kwargs):
        return self.query_vector('Incentive', netuid=netuid,  block=block, update=update, **kwargs)
    incentive = incentives

    def last_update(self, netuid = 0, update=False, **kwargs):
        return self.query_vector('LastUpdate', netuid=netuid,   update=update, **kwargs)

    def dividends(self, netuid = 0, update=False, **kwargs):
        return  self.query_vector('Dividends', netuid=netuid,   update=update,  **kwargs)
            
    dividend = dividends
    
    def names(self, 
              netuid: int = 0, 
              update=False,
                **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
        names = self.query_map('Name', update=update, netuid=netuid,**kwargs)
        if netuid == 'all':
            for netuid, netuid_names in names.items():
                names[netuid] = list(netuid_names.values())
        else:
            names = list(names.values())
        return names

    def uids(self,
             netuid = 0,
              update=False, 
              max_age=1000,
             **kwargs) -> List[str]:
        netuid = self.resolve_netuid(netuid)
        keys =  self.query_map('Keys', netuid=netuid, update=update,  max_age=max_age, **kwargs)
        if netuid == 'all':
            for netuid, netuid_keys in keys.items():
                keys[netuid] = list(netuid_keys.keys ())
        else:
            keys = list(keys.keys())
        return keys
    
    def subnet2n(self, fmt='j',  **kwargs):
        netuid2n = self.netuid2n(fmt=fmt, **kwargs)
        netuid2subnet = self.netuid2subnet()
        subnet2n = {}
        for netuid, subnet in netuid2subnet.items():
            subnet2n[subnet] = netuid2n[netuid]
        return subnet2n
    
    def subnet2stakes(self,  block=None, update=False, fmt='j', **kwargs):
        subnet2stakes = {}
        for netuid in self.netuids( update=update):
            subnet2stakes[netuid] = self.stakes(netuid=netuid,  block=block, update=update, fmt=fmt, **kwargs)
        return subnet2stakes

    def subnet_state(self,  netuid='all', block=None, update=False, fmt='j', **kwargs):

        subnet_state = {
            'params': self.subnet_params(netuid=netuid,  block=block, update=update, fmt=fmt, **kwargs),
            'modules': self.modules(netuid=netuid,  block=block, update=update, fmt=fmt, **kwargs),
        }
        return subnet_state

    def subnet2emission(self, fmt='j',  **kwargs):
        subnet2params = self.subnet_params(netuid='all')
        netuid2emission = self.netuid2emission(fmt=fmt, **kwargs)
        netuid2subnet = self.netuid2subnet()
        subnet2emission = {}
        for netuid, subnet in netuid2subnet.items():
            subnet2emission[subnet] = netuid2emission[netuid]
        # sort by emission
        subnet2emission = dict(sorted(subnet2emission.items(), key=lambda x: x[1], reverse=True))

        return subnet2emission


    def uid2key(self, uid=None, 
             netuid = 0,
              update=False, 
              
             max_age= 1000,
             **kwargs):
        netuid = self.resolve_netuid(netuid)
        uid2key =  self.query_map('Keys',  netuid=netuid, update=update,  max_age=max_age, **kwargs)
        # sort by uid
        if uid != None:
            return uid2key[uid]
        return uid2key
    

    def key2uid(self, key = None, netuid: int = 0, update=False, netuids=None , **kwargs):
        uid2key =  self.uid2key( netuid=netuid, update=update, **kwargs)
        reverse_map = lambda x: {v: k for k,v in x.items()}
        if netuid == 'all':
            key2uid =  {netuid: reverse_map(_key2uid) for netuid, _key2uid in uid2key.items()  if   netuids == None or netuid in netuids  }
        else:
            key2uid = reverse_map(uid2key)
        if key != None:
            key_ss58 = self.resolve_key_ss58(key)
            return key2uid[key_ss58]
        return key2uid

    """ Returns network SubnetN hyper parameter """
    def n(self,  netuid: int = 0,block: Optional[int] = None, max_age=100, update=False, **kwargs ) -> int:
        if netuid == 'all':
            return sum(self.query_map('N', block=block , update=update, max_age=max_age,  **kwargs).values())
        else:
            return self.query( 'N', params=[netuid], block=block , update=update,  **kwargs)

    def subnet_exists(self, subnet:str) -> bool:
        subnets = self.subnets()
        return bool(subnet in subnets)

    def subnet_emission(self, netuid:str = 0, block=None, update=False, **kwargs):
        emissions = self.emission(block=block, update=update,  netuid=netuid, **kwargs)
        if isinstance(emissions[0], list):
            emissions = [sum(e) for e in emissions]
        return sum(emissions)
    
    def get_modules(self,
                    keys : list = None,
                    netuid=0,
                    timeout=30,
                    min_emission=0,
                    max_age = 1000,
                    update=False,
                    **kwargs):
        modules = None
        path = None 
        if keys == None :
            path = f'subnet/{self.network}/{netuid}/modules'
            modules = self.get(path, None, max_age=max_age, update=update)
            keys = self.keys(netuid=netuid)

        n = len(keys)
        if modules == None:
            modules = []
            print(f'Getting modules {n}')
            futures = [c.submit(self.get_module, kwargs=dict(module=k, netuid=netuid, **kwargs)) for k in keys]
            progress = c.tqdm(n)
            modules = []


            should_pass = lambda x: isinstance(x, dict) \
                            and 'name' in x \
                            and len(x['name']) > 0 \
                            and x['emission'] >= min_emission
            
            for future in c.as_completed(futures, timeout=timeout):
                module = future.result()
                if should_pass(module):
                    modules += [module]
                    progress.update(1)
            if path != None:
                self.put(path, modules)
            
                    
        return modules

    module_param_features = [
        'key',
        'name',
        'address',
        'emission',
        'incentive',
        'dividends',
        'last_update',
        'stake_from',
        'delegation_fee'
    ]
    
    def get_module(self, 
                    module=None,
                    netuid=0,
                    trials = 4,
                    fmt='j',
                    mode = 'http',
                    block = None,
                    max_age = None,
                    lite = True, 
                    update = False,
                    **kwargs ) -> 'ModuleInfo':
        U16_MAX = 2**16 - 1
        
        if module == None:
            module = self.keys(netuid=netuid, update=update, max_age=max_age)[0]
            c.print(f'No module specified, using {module}')
        
        module = c.key2address().get(module, module)
        url = self.resolve_url( mode=mode)
        module_key = module
        is_valid_key = c.valid_ss58_address(module)
        if not is_valid_key:
            module_key = self.name2key(name=module,  netuid=netuid, **kwargs)
        netuid = self.resolve_netuid(netuid)
        json={'id':1, 'jsonrpc':'2.0',  'method': 'subspace_getModuleInfo', 'params': [module_key, netuid]}
        module = None
        for i in range(trials):
            try:
                module = requests.post(url,  json=json).json()
                break
            except Exception as e:
                c.print(e)
                continue
        assert module != None, f"Failed to get module {module_key} after {trials} trials"
        if not 'result' in module:
            return module
        module = {**module['result']['stats'], **module['result']['params']}
        # convert list of u8 into a string Vector<u8> to a string
        module['name'] = self.vec82str(module['name'])
        module['address'] = self.vec82str(module['address'])
        module['dividends'] = module['dividends'] / (U16_MAX)
        module['incentive'] = module['incentive'] / (U16_MAX)
        module['stake_from'] = {k:self.format_amount(v, fmt=fmt) for k,v in module['stake_from']}
        module['stake'] = sum([v for k,v in module['stake_from'].items() ])
        module['emission'] = self.format_amount(module['emission'], fmt=fmt)
        module['key'] = module.pop('controller', None)
        module['metadata'] = module.pop('metadata', {})

        module['vote_staleness'] = (block or self.block) - module['last_update']
        if lite :
            features = self.module_param_features + ['stake', 'vote_staleness']
            module = {f: module[f] for f in features}
        assert module['key'] == module_key, f"Key mismatch {module['key']} != {module_key}"
        return module


    def root_valis(self, netuid = 0, update=False, **kwargs):
        return self.get_modules(netuid=netuid, update=update, **kwargs)




    def root_keys(self, netuid = 0, update=False, **kwargs):
        return self.keys(netuid=netuid, update=update, **kwargs)




    def registration_block(self, netuid: int = 0, update=False, **kwargs):
        registration_blocks = self.query_map('RegistrationBlock', netuid=netuid, update=update, **kwargs)
        return registration_blocks

    regblocks = registration_blocks = registration_block





    

    def key2name(self, key=None, netuid: int = None, update=False) -> Dict[str, str]:
        
        key2name =  {v:k for k,v in self.name2key(netuid=netuid,  update=update).items()}
        if key != None:
            return key2name[key]
        return key2name
        



    def name2key(self, name:str=None, 
                 max_age=1000, 
                 timeout=30, 
                 netuid: int = 0, 
                 update=False, 
                 trials=3,
                 **kwargs ) -> Dict[str, str]:
        # netuid = self.resolve_netuid(netuid)
        netuid = self.resolve_netuid(netuid)

        names = c.submit(self.names, kwargs={'feature': 'names', 'netuid':netuid, 'update':update, 'max_age':max_age, 'network': self.network})
        keys = c.submit(self.keys, kwargs={'feature': 'keys', 'netuid':netuid, 'update':update, 'max_age':max_age, 'network': self.network})
        names, keys = c.wait([names, keys], timeout=timeout)
        name2key = dict(zip(names, keys))
        if name != None:
            if name in name2key:
                return name2key[name]
            else:
                trials -= 1
                if trials == 0:
                    return None
                else:
                    return self.name2key(name=name,
                                        timeout=timeout, netuid=netuid, update=True, 
                                        trials=trials, **kwargs)
                
        return name2key



              
    def name2uid(self, name = None, netuid: int = 0, search=None) -> int:
        netuid = self.resolve_netuid(netuid)
        uid2name = self.uid2name(netuid=netuid)

        if netuid == 'all':
            netuid2name2uid = {}
            for netuid, netuid_uid2name in uid2name.items():
                name2uid = self.search_dict(netuid_uid2name)
                if name != None:
                    name2uid = name2uid[name] 
                netuid2name2uid[netuid] = name2uid
            return netuid2name2uid
            
        else:
            name2uid = {v:k for k,v in uid2name.items()}
            if search != None:
                name2uid =  self.search_dict(name2uid, search=search)
            if name != None:
                return name2uid[name] 
            
        return name2uid

    def netuids(self,  update=False, block=None) -> Dict[int, str]:
        return list(self.netuid2subnet( update=update, block=block).keys())

    def netuid2subnet(self, netuid=None,  update=False, block=None, **kwargs ) -> Dict[str, str]:
        netuid2subnet = self.query_map('SubnetNames', update=update,  block=block, **kwargs)
        netuid2subnet = dict(sorted(netuid2subnet.items(), key=lambda x: x[0]))
        if netuid != None:
            return netuid2subnet[netuid]
        return netuid2subnet
    
    def subnet2netuid(self, subnet=None,  update=False,  **kwargs ) -> Dict[str, str]:
        subnet2netuid =  {v:k for k,v in self.netuid2subnet( update=update, **kwargs).items()}
        # sort by subnet 
        if subnet != None:
            return subnet2netuid[subnet] if subnet in subnet2netuid else len(subnet2netuid)
        return subnet2netuid

    
    
    def get_uid( self, key: str, netuid: int = 0, block: Optional[int] = None, update=False, **kwargs) -> int:
        return self.query( 'Uids', block=block, params=[ netuid, key ] , update=update, **kwargs)  


    def weights(self,  netuid = 0,  update=False, **kwargs) -> list:
        weights =  self.query_map('Weights',netuid=netuid, update=update, **kwargs)

        tuples2list = lambda x: [list(v) for v in x]
        if netuid == 'all':
            for netuid, netuid_weights in weights.items():
                weights[netuid] = {k: tuples2list(v) for k,v in netuid_weights.items()}
        else:
            weights = {k: tuples2list(v) for k,v in weights.items()}
            
        return weights
    




    def total_emissions(self, netuid = 9, block=None, update=False, fmt = 'j', **kwargs):

        emissions = self.query_vector('Emission',  netuid=netuid, block=block, update=update, **kwargs)
        if netuid == 'all':
            for netuid, netuid_emissions in emissions.items():
                emissions[netuid] = [self.format_amount(e, fmt=fmt) for e in netuid_emissions]
        else:
            emissions = [self.format_amount(e, fmt=fmt) for e in emissions]
        
        return sum(emissions)
    


    # def state(self, block=None, netuid='all', update=False, max_age=10000, fmt='j', **kwargs):
    #     subnet_params = self.subnet_params(block=block, netuid=netuid, max_age=max_age, **kwargs)
    #     subnet2emissions = self.emissions(netuid=netuid, max_age=max_age, block=block, **kwargs)
    #     subnet2staketo = self.stake_to(netuid=netuid, block=block, update=update, fmt=fmt, **kwargs)
    #     subnet2incentives = self.incentives(netuid=netuid, block=block, update=update, fmt=fmt, **kwargs)
    #     subnet2trust = self.trust(netuid=netuid, block=block, update=update, fmt=fmt, **kwargs)
    #     subnet2keys = self.keys(netuid=netuid, block=block, update=update, **kwargs)
                        
    #     subnet2state = {}
    #     for netuid, params in subnet_params.items():
    #         subnet_state = {
    #             'params': params,
    #             'incentives': subnet2incentives[netuid],
    #             'emissions': subnet2emissions[netuid],
    #             ''
    #             'stake_to': subnet2staketo[netuid],
    #             'keys': subnet2keys[netuid],

    #         }
    #         subnet_state[netuid] = subnet_state
        
    #     return subnet2state


    def netuid2emission(self, fmt='j',  period='day', **kwargs):
        netuid2emission = {}
        netuid2tempo = None
        emissions = self.query_vector('Emission',  netuid='all', **kwargs)
        for netuid, netuid_emissions in emissions.items():
            
            if period == 'day':
                if netuid2tempo == None:
                    netuid2tempo = self.query_map('Tempo', netuid='all', **kwargs)
                tempo = netuid2tempo.get(netuid, 100)
                multiplier = self.blocks_per_day() / tempo
            else:
                multiplier = 1
            netuid2emission[netuid] = self.format_amount(sum(netuid_emissions), fmt=fmt) * multiplier
        
        netuid2emission = {k: v   for k,v in netuid2emission.items()}

        return netuid2emission

    def subnet2emission(self, fmt='j',  period='day', **kwargs):
        netuid2emission = self.netuid2emission(fmt=fmt, period=period, **kwargs)
        netuid2subnet = self.netuid2subnet()
        subnet2emission = {}
        for netuid, emission in netuid2emission.items():
            subnet = netuid2subnet[netuid]
            subnet2emission[subnet] = emission
        return subnet2emission
    

    def global_emissions(self,  **kwargs):
        return sum(list(self.subnet2emissions( **kwargs).values()))


    

    def subnet2params( self,  block: Optional[int] = None ) -> Optional[float]:
        netuids = self.netuids()
        subnet2params = {}
        netuid2subnet = self.netuid2subnet()
        for netuid in netuids:
            subnet = netuid2subnet[netuid]
            subnet2params[subnet] = self.subnet_params(netuid=netuid, block=block)
        return subnet2params
    



    #################
    #### UPDATE SUBNET ####
    #################
    def update_subnet(
        self,
        params: dict= None,
        netuid: int = 0,
        key: str = None,
        nonce = None,
        update= True,
        **extra_params,
    ) -> bool:
        
        params = {**(params or {}), **extra_params}
            
        netuid = self.resolve_netuid(netuid)
        subnet_params = self.subnet_params( netuid=netuid , update=update, fmt='nanos')
        # infer the key if you have it
        for k in ['min_stake']:
            if k in params:
                params[k] = params[k] * 1e9
        if key == None:
            key2address = self.address2key()
            if subnet_params['founder'] not in key2address:
                return {'success': False, 'message': f"Subnet {netuid} not found in local namespace, please deploy it "}
            key = c.get_key(key2address.get(subnet_params['founder']))
            c.print(f'Using key: {key}')

        # remove the params that are the same as the module info
        params = {**subnet_params, **params}
        for k in ['name']:
            params[k] = params[k].encode('utf-8')
        params['netuid'] = netuid
        return self.compose_call(fn='update_subnet',
                                     params=params, 
                                     key=key, 
                                     nonce=nonce)


    #################
    #### Serving ####
    #################
    def propose_subnet_update(
        self,
        netuid: int = None,
        key: str = None,
        nonce = None,
        **params,
    ) -> bool:

        netuid = self.resolve_netuid(netuid)
        c.print(f'Adding proposal to subnet {netuid}')
        subnet_params = self.subnet_params( netuid=netuid , update=True)
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



    def get_stake( self, key_ss58: str, block: Optional[int] = None, netuid:int = None , fmt='j', update=True ) -> Optional['Balance']:
        
        key_ss58 = self.resolve_key_ss58( key_ss58)
        netuid = self.resolve_netuid( netuid )
        stake = self.query( 'Stake',params=[netuid, key_ss58], block=block , update=update)
        return self.format_amount(stake, fmt=fmt)



    def min_register_stake(self, netuid: int = 0, fmt='j', **kwargs) -> float:
        netuid = self.resolve_netuid(netuid)
        min_burn = self.min_burn(  fmt=fmt)
        min_stake = self.min_stake(netuid=netuid,  fmt=fmt)
        return min_stake + min_burn
    



    def resolve_netuid(self, netuid: int = None) -> int:
        '''
        Resolves a netuid to a subnet name.
        '''
        if netuid == 'all':
            return netuid
        if netuid == None :
            # If the netuid is not specified, use the default.
            return 0
        if isinstance(netuid, str):
            subnet2netuid = self.subnet2netuid()
            if netuid not in subnet2netuid: # if still not found, try lower case
                subnet2netuid =self.subnet2netuid(update=True)
            assert netuid in subnet2netuid, f"Subnet {netuid} not found in {subnet2netuid}"
            return subnet2netuid[netuid]

        elif isinstance(netuid, int):
            if netuid == 0: 
                return netuid
            # If the netuid is an integer, ensure it is valid.
            
        assert isinstance(netuid, int), "netuid must be an integer"
        return netuid
    

    def blocks_until_vote(self, netuid=0, **kwargs):
        netuid = self.resolve_netuid(netuid)
        tempo = self.subnet_params(netuid=netuid, **kwargs)['tempo']
        block = self.block
        return tempo - ((block + netuid) % tempo)

    def emission_per_epoch(self, netuid=None):
        return self.subnet(netuid=netuid)['emission']*self.epoch_time(netuid=netuid)



    

    def get_stake_to( self, 
                     key: str = None, 
                     module_key=None,
                       block: Optional[int] = None, 
                        fmt='j' , update=False,
                        max_age = 60,
                        timeout = 10,
                         **kwargs) -> Optional['Balance']:
        
        key_address = self.resolve_key_ss58( key )
        stake_to = self.query_map( 'StakeTo', params=[ key_address], block=block, update=update,  max_age=max_age)
        stake_to =  {k: self.format_amount(v, fmt=fmt) for k, v in stake_to.items()}
        return stake_to
    
    def get_stake_from( self, key: str, block: Optional[int] = None,  fmt='j', update=True  ) -> Optional['Balance']:
        key = self.resolve_key_ss58( key )
        stake_from = self.query_map( 'StakeFrom', params=[key], block=block,  update=update )
        stake_from =  {k: self.format_amount(v, fmt=fmt) for k, v in stake_from.items()}
        return stake_from


    def epoch_time(self, netuid=0, update=False, **kwargs):
        return self.subnet_params(netuid=netuid, update=update, **kwargs)['tempo']*self.block_time

    def seconds_per_day(self, ):
        return 24*60*60
    
    def epochs_per_day(self, netuid=0):
        return self.seconds_per_day()/self.epoch_time(netuid=netuid)

    def seconds_per_epoch(self, netuid='all'):
        netuid =self.resolve_netuid(netuid)
        return self.block_time * self.subnet_params(netuid=netuid)['tempo']


    def format_module(self, module: 'ModuleInfo', fmt:str='j') -> 'ModuleInfo':
        U16_MAX = 2**16 - 1
        for k in ['emission']:
            module[k] = self.format_amount(module[k], fmt=fmt)
        for k in ['incentive', 'dividends']:
            module[k] = module[k] / (U16_MAX)
        
        module['stake_from'] = {k: self.format_amount(v, fmt=fmt)  for k, v in module['stake_from']}
        return module
