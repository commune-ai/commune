import commune as c

class Ticket(c.Module):
    seperator='<DATA::SIGNATURE>'
    def create(self, key, seperator=seperator, tag=None, **kwargs):
        timestamp = str(c.time())
        key = c.get_key(key)
        c.print(key)
        if tag:
            timestamp += f'::{tag}'
        return key.sign(timestamp, return_string=True, seperator=seperator)

    @classmethod
    def test(cls, key='test'):
        self = cls()
        ticket = self.create(key)
        key = self.key
        c.print(ticket)
        assert c.verify(ticket, key=key)
        ticket2signer = self.ticket2signer(ticket)
        c.print(ticket2signer)
        assert ticket2signer == key.ss58_address
        return {'ticket': ticket, 'key': key.ss58_address}

    def ticket2signer(self, ticket = None, seperator=seperator, **kwargs):
        
        return c.verify(ticket, return_address=True, seperator=seperator, **kwargs)
