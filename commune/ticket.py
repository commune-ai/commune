import commune as c

class Ticket(c.Module):
    seperator='"<DATA::SIGNATURE>"'
    def create(self, key, seperator=seperator):
        timestamp = str(c.time())
        key = c.get_key(key)
        return key.sign(timestamp, return_str=True, seperator=seperator)

    def verify(self, ticket, key, seperator=seperator):
        timestamp, signature = ticket.split(seperator)
        key = c.get_key(key)
        return key.verify(data=timestamp, signature=signature, seperator=seperator)

    @classmethod
    def test(cls, key='test'):
        c.print('fam')
        key = c.get_key(key)
        self = cls()
        ticket = self.create(key)
        c.print(ticket)
        assert c.verify(ticket, key=key, seperator=self.seperator)
        return {'ticket': ticket, 'key': key.ss58}

    def signature2signer(self, signature = None, key=None, seperator=seperator, **kwargs):
        if signature == None:
            key = c.get_key(key)
            signature = key.ticket()
        return c.verify(signature=signature, return_address=True, seperator=seperator, **kwargs)
