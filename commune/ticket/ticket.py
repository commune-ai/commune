import commune as c

class Ticket(c.Module):
    seperator='<DATA::SIGNATURE>'
    def create(self, key, seperator=seperator, tag=None, **kwargs):
        timestamp = str(c.time())
        if tag:
            timestamp += f'::{tag}'
        return key.sign(timestamp, return_string=True, seperator=seperator)

    def verify(self, ticket, seperator=seperator, return_address=True, max_age=100, **kwargs):
        timestamp = c.time()
        assert c.verify(ticket, seperator=seperator, return_address=return_address,  **kwargs)
        staleness = timestamp - int(ticket.split(seperator)[0])
        if staleness > max_age:
            return False
        return True

    @classmethod
    def test(cls, key='test'):
        key = c.get_key(key)
        self = cls()
        ticket = self.create(key)
        key = self.key
        assert self.verify(ticket, key=key, seperator=self.seperator)
        ticket2signer = self.ticket2signer(ticket)
        assert ticket2signer == key.ss58_address
        return {'ticket': ticket, 'key': key.ss58_address}

    def ticket2signer(self, ticket = None, seperator=seperator, **kwargs):
        
        return c.verify(ticket, return_address=True, seperator=seperator, **kwargs)
