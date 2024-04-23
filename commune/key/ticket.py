import commune as c

class Ticket(c.Module):

    seperator='::ticket::'

    def create(self, key, seperator=seperator):
        timestamp = str(c.time())
        key = c.get_key(key)
        return key.sign(timestamp, return_string=True, seperator=seperator)

    def verify(self, ticket, key, seperator=seperator):
        timestamp, signature = ticket.split(seperator)
        key = c.get_key(key)
        return key.verify(data=timestamp, signature=signature, seperator=seperator)

    def test(self):
        key = c.get_key('test')
        ticket = self.create(key)
        c.print(ticket)
        assert self.verify(ticket, key, seperator=self.seperator)
        return {'ticket': ticket, 'key': str(key)}

    



        
