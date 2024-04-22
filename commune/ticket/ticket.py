import commune as c

class Ticket(c.Module):
    seperator='<DATA::SIGNATURE>'
    def create(self, key, seperator=seperator, tag=None, **kwargs):
        timestamp = str(c.time())
        if tag:
            timestamp += f'::{tag}'
        return key.sign(timestamp, return_string=True, seperator=seperator)

    def verify(self, ticket, key, seperator=seperator, max_age=100):
        data, signature = ticket.split(seperator)
        key = c.get_key(key)
        signature = key.verify(data=data, signature=signature, seperator=seperator)

        if '::' in data:
            timestamp, tag = data.split('::')
        else:
            timestamp, tag = data, None
        timestamp = float(timestamp)
        if c.time() - timestamp > max_age:
            return False
        
        return True

    @classmethod
    def test(cls, key='test'):
        c.print('fam')
        key = c.get_key(key)
        self = cls()
        ticket = self.create(key)
        key = self.key
        assert self.verify(ticket, key=key, seperator=self.seperator)
        return {'ticket': ticket, 'key': key.ss58_address}

    def signature2signer(self, signature = None, key=None, seperator=seperator, **kwargs):
        if signature == None:
            key = c.get_key(key)
            signature = key.ticket()
        return c.verify(signature=signature, return_address=True, seperator=seperator, **kwargs)
