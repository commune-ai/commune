import commune as c

class ValiParity(c.Module):
    def __init__(self, run=True):
        self.subspace = c.module('subspace')()
        self.subnet = self.subspace.subnet()
        if run:
            c.thread(self.run)
        self.seconds_per_epoch = self.subnet['tempo'] * 8

    def votes(self, max_trust = 25) -> int:
        modules = self.subspace.modules()
        voted_modules = c.shuffle([m for m in modules if m['trust'] < max_trust])[:self.subnet['max_allowed_weights']]
        uids = [m['uid'] for m in voted_modules]
        max_trust = max([m['trust'] for m in voted_modules])
        weights = [max_trust - m['trust'] for m in voted_modules]
        return {'uids': uids, 'weights': weights}
    

    def run(self):
        while True:
            c.print('voting...')
            r = self.vote()
            c.print(r)
            self.sleep(self.seconds_per_epoch)
    def vote(self, key=None):
        key = self.resolve_key(key)
        try:
            votes = self.votes()
            response = self.subspace.vote(**votes, key=key)
        except Exception as e:
            self.subspace = c.module('subspace')()
            e = c.detailed_error(e)
            c.print(e)
        return response

    @classmethod
    def regloop(cls, n=100, tag='commie', remote: str = True, key=None, timeout=30):

        if remote:
            kwargs = c.locals2kwargs(locals())
            kwargs['remote'] = False
            return cls.remote_fn('regloop', kwargs=kwargs)



        cnt = 0
        self = cls(run=False)

        while True:
            registered_servers = []
            subspace = c.module('subspace')()
            namespace =  subspace.namespace(search=self.module_path())

            c.print('registered servers', namespace)
            ip = c.ip()
            i = 0
            name2futures = {}
            while cnt < n:

                name = cls.resolve_server_name(tag=tag+str(i))
                module_key = c.get_key(name)

                futures = name2futures.get(name, [])

                address = ip + ':' + str(30333 + cnt)

                if name in namespace:
                    i += 1
                    c.print('already registered', name)
                    continue
                
                try:
                    c.print('registering', name)
                    response = self.subspace.register(name=name, address=address, module_key=module_key.ss58_address, key=key)
                    if not response['success']:
                        if response['error']['name'] == 'NameAlreadyRegistered':
                            i += 1
                    c.print(response)
                except Exception as e:
                    e = c.detailed_error(e)
                    c.print(e)

    @classmethod
    def voteloop(cls, remote: str = True, min_staleness=200, search='vali', namespace=None):
        if remote:
            kwargs = c.locals2kwargs(locals())
            kwargs['remote'] = False
            return cls.remote_fn('voteloop', kwargs=kwargs)

        self = cls(run=False)
        stats = c.stats(search=search, df=False)
        for module in stats:
            c.print('voting for', module['name'])
            if module['last_update'] > min_staleness:
                c.print('skipping', module['name'], 'because it is was voted recently')

            self.vote(key=module['name'])

        for name in namespace:
            c.print('voting for', name)
            self.vote(key=name)

        
                    

    