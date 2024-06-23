import commune as c
import json

class Ticket(c.Module):
    description = """
    # THIS CREATES A TOKEN THAT CAN BE USED TO VERIFY THE ORIGIN OF A MESSAGE, AND IS GENERATED CLIENT SIDE
    # THIS USES THE SAME T
    # ECHNOLOGY AS ACCESS TOKENS, BUT IS USED FOR CLIENT SIDE VERIFICATION, AND NOT SERVER SIDE
    # THIS GIVES USERS THE ABILITY TO VERIFY THE ORIGIN OF A MESSAGE, AND TO VERIFY THAT THE MESSAGE HAS NOT BEEN TAMPERED WITH
    #data={DATA}::address={ADDRESS}::time={time}::signature={SIGNATURE}
    where variable_seperator = '::'
    """
    variable_seperator = '::'
    signature_seperator = variable_seperator + 'signature='

    def create(self, data=None, key=None, json_str=False, **kwargs):
        """
        params:
            data: dict: data to be signed
            key: str: key to sign with
            json_str: bool: if True, the ticket will be returned as a json string
            
        """
        key = c.get_key(key)
        ticket_dict = {
            'data': data,
            'time': c.time(),
            'address': key.ss58_address,
        }
        data = self.dict2ticket(ticket_dict, json_str=json_str)
        ticket = key.sign(data, 
                          return_string=True, 
                          seperator=self.signature_seperator, **kwargs)
        return ticket

    ticket = create
    
    def dict2ticket(self, ticket, json_str=False):
        """
        Convert a dictionary to a ticket string
        """
        if json_str:
            return json.dumps(ticket)
        else:
            ticket_str = ''
            for i, (k,v) in enumerate(ticket.items()):
                ticket_str +=  ((self.variable_seperator if i > 0 else "") +k + '=' + str(v) )

        return ticket_str

    def ticket2address(self, ticket):
        """
        Get the address from a ticket
        """
        ticket_dict = self.ticket2dict(ticket)
        return ticket_dict['address']

    def ticket2dict(self, ticket):
        """
        Convert a ticket string to a dictionary
        """
        if ticket.startswith('{'):
            ticket_splits = ticket.split(self.signature_seperator)
            signature = ticket_splits[1]
            ticket = ticket_splits[0]
            c.print(ticket, 'ticket')
            ticket_dict = json.loads(ticket)
            ticket_dict['signature'] = signature
        else:
            ticket_dict = {}
            for item in ticket.split(self.variable_seperator):
                k,v = item.split('=')
                ticket_dict[k] = v
            ticket_dict['time'] = float(ticket_dict['time'])
            
        return ticket_dict
    
    
    def verify(self, ticket,  max_age:str=5,  age=None, timeout=None, **kwargs):
        max_age = age or timeout or max_age 
        ticket_dict = self.ticket2dict(ticket)
        address = ticket_dict['address']
        staleness = c.time() - ticket_dict['time']
        if staleness > max_age:
            return False
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
    def test_staleness(cls, key='test', max_age=0.5):
        c.add_key(key)
        key = c.get_key(key)
        self = cls()
        ticket = self.create(key=key)
        assert self.ticket2address(ticket) == key.ss58_address, f"{self.ticket2address(ticket)} != {key.ss58_address}"
        print('waiting for staleness')
        c.sleep(max_age + 0.1)
        key = c.get_key(key)
        assert not self.verify(ticket, max_age=max_age), 'Failed to verify'
        return {'success': True, 'ticket': ticket, 'key': str(key)}

    def qr(self,filename='ticket.png'):
        return c.module('qrcode').text2qrcode(self.ticket(), filename=filename)