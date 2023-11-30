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
            e = c.detailed_error(e)
            c.print(e)
        return response

    @classmethod
    def regloop(cls, n=100, tag='commie', remote: str = False):
        if remote:
            kwargs = c.locals2kwargs(locals())
            kwargs['remote'] = False
            return cls.remote_fn('regloop', kwargs=kwargs)


        subspace = c.module('subspace')()

        while True:
            namespace = subspace.namespace(update=True)
            ip = c.ip()
            for i in range(n):

                name = cls.resolve_server_name(tag=tag+str(i))
                key = c.get_key(name)
                address = ip + ':' + str(30333 + i)
                
                try:
                    response = subspace.register(name=name, address=address, key=key)
                    if response['success']:
                        cls().vote()
                
                except Exception as e:
                    c.print(e)
                    self.sleep(10)

    