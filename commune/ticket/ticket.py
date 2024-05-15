import commune as c

class Ticket(c.Module):
    seperator='<SIGNATURE>'
    def create(self, key=None, seperator=seperator, tag=None, **kwargs):
        key = c.get_key(key) if key else self.key
        timestamp = str(c.time())
        if tag:
            timestamp += f'::{tag}'
        return key.sign(timestamp, return_string=True, seperator=seperator) + '<ADDRESS>' + key.ss58_address
    
    def verify(self, ticket, seperator=seperator):
        assert seperator in ticket, f'No seperator found in {ticket}'
        ticket, address = ticket.split('<ADDRESS>')
        return c.verify(ticket, address=address, seperator=seperator)

    @classmethod
    def test(cls, key='test'):
        self = cls()
        ticket = self.create(key)
        key = c.get_key(key)
        c.print(ticket)
        assert self.verify(ticket), 'Failed to verify'
        return {'success': True, 'ticket': ticket, 'key': str(key)}

    def qr(self,filename='ticket.png'):
        return c.module('qrcode').text2qrcode(self.ticket(), filename=filename)