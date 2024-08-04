import commune as c
import json

class Ticket(c.Module):
    ticket_features = ['signature', 'address', 'crypto_type']
    data_features = ['data', 'time']
    max_age = 10
    description = """
    {
        'data': dict (SIGNED)
        'time': int: (SIGNED)
        'signature': str (NOT SIGNED): the signature of the data
        'address': str: (NOT SIGNED): the address of the signer
        'crypto_type': str/int: (NOT SIGNED): the type of crypto used to sign
    }

    To verify 

    """

    def ticket(self, data='commune', key=None, **kwargs):
        """
        params:
            data: dict: data to be signed
            key: str: key to sign with
            json_str: bool: if True, the ticket will be returned as a json string
            
        """
        key = c.get_key(key) if key else self.key
        ticket_dict = {
            'data': data,
            'time': c.time(),
        }
        signtature = key.sign(ticket_dict, **kwargs).hex()
        ticket_dict.update({'signature': signtature, 
                            'address': key.ss58_address,
                            'crypto_type': key.crypto_type})
        return ticket_dict

    create = ticket
    
    def ticket2address(self, ticket):
        """
        Get the address from a ticket
        """
        return ticket['address']
    

    def is_ticket_dict(self, ticket):
        if isinstance(ticket, dict):
            return all([self.is_ticket(v) in ticket for v in ticket.values()])
        return False
    

    def is_ticket_list(self, ticket):
        if isinstance(ticket, list):
            return all([self.is_ticket(v) in ticket for v in ticket])
        return False


    def is_ticket(self, data):
        return all([f in data for f in self.ticket_features])

    def verify(self, data, 
                max_age:str=None,  
               **kwargs):
        data = c.copy(data)
        max_age = max_age or self.max_age 
        date_time = data.get('time', data.get('timestamp'))
        staleness = c.time() - date_time
        if staleness >  max_age:
            print(f"Signature too Old! from {data} : {staleness} > {max_age}")
            return False

        ticket = {}
        if 'ticket' in data:
            ticket = data.pop('ticket')
        elif self.is_ticket(data):
            for f in self.ticket_features:
                ticket[f] = data.pop(f)
        else:
            raise ValueError(f"Data is not a ticket: {data}")
        return c.verify(data, **ticket)


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
        assert not c.verify_ticket(ticket, max_age=max_age), 'Failed to verify'
        return {'success': True, 'ticket': ticket, 'key': str(key)}

    def qr(self,filename='./ticket.png'):
        filename = self.resolve_path(filename)
        return c.module('qrcode').text2qrcode(self.ticket(), filename=filename)
    


    @classmethod
    def test(cls, key='test'):
        key = c.new_key()
        self = cls()
        ticket = self.ticket(key=key)
        reciept = self.verify(ticket)
        print(reciept)
        assert reciept, 'Failed to verify'
        return {'success': True, 'ticket': ticket, 'key': str(key), 'reciept': reciept}
    