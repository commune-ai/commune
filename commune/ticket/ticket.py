import commune as c
import json

class Ticket(c.Module):
    description = """
    # THIS CREATES A TOKEN THAT CAN BE USED TO VERIFY THE ORIGIN OF A MESSAGE, AND IS GENERATED CLIENT SIDE
    # THIS USES THE SAME TECHNOLOGY AS ACCESS TOKENS, BUT IS USED FOR CLIENT SIDE VERIFICATION, AND NOT SERVER SIDE
    # THIS GIVES USERS THE ABILITY TO VERIFY THE ORIGIN OF A MESSAGE, AND TO VERIFY THAT THE MESSAGE HAS NOT BEEN TAMPERED WITH
    #data={DATA}::address={ADDRESS}::time={time}::signature={SIGNATURE}
    """
    signature_seperator = '::signature='

    def create(self, data=None, key=None, **kwargs):
        key = c.get_key(key)
        ticket_dict = {
            'data': data,
            'time': c.time(),
            'address': key.ss58_address,
        }
        data = self.dict2ticket(ticket_dict)
        ticket = key.sign(data, return_string=True, seperator=self.signature_seperator)
        return ticket
    
    def dict2ticket(self, ticket):
        """
        Convert a dictionary to a ticket string
        """
        ticket_str = ''
        for i, (k,v) in enumerate(ticket.items()):
            ticket_str +=  (("::" if i > 0 else "") +k + '=' + str(v) )
        return ticket_str

    def ticket2dict(self, ticket):
        """
        Convert a ticket string to a dictionary
        """
        if ticket.startswith('{'):
            ticket = json.loads(ticket)
        if isinstance(ticket, str):
            ticket_dict = {}
            for item in ticket.split('::'):
                k,v = item.split('=')
                ticket_dict[k] = v
            ticket_dict['time'] = float(ticket_dict['time'])
        return ticket_dict
    
    
    def verify(self, ticket,  max_age:str=5, **kwargs):
        ticket_dict = self.ticket2dict(ticket)
        address = ticket_dict['address']
        staleness = c.time() - ticket_dict['time']
        c.print(staleness)

        if staleness > max_age:
            return False
        c.print(ticket)
        return c.verify(ticket, address=address, seperator=self.signature_seperator,  **kwargs)

    @classmethod
    def test_verification(cls, key='test'):
        c.add_key(key)
        key = c.get_key(key)
        self = cls()
        ticket = self.create(key=key)
        key = c.get_key(key)
        c.print(ticket)
        reciept = self.verify(ticket)
        assert reciept, 'Failed to verify'
        return {'success': True, 'ticket': ticket, 'key': str(key), 'reciept': reciept}
    

    @classmethod
    def test_staleness(cls, key='test', max_age=1):
        c.add_key(key)
        key = c.get_key(key)
        self = cls()
        ticket = self.create()
        print('waiting for staleness')
        c.sleep(max_age + 1)
        key = c.get_key(key)
        c.print(ticket)
        reciept = self.verify(ticket, max_age=max_age)
        assert not reciept, 'Failed to verify'
        return {'success': True, 'ticket': ticket, 'key': str(key), 'reciept': reciept}

    def qr(self,filename='ticket.png'):
        return c.module('qrcode').text2qrcode(self.ticket(), filename=filename)