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
        }
        signtature = key.sign(ticket_dict, **kwargs).hex()
        ticket = {'signature': signtature, 'address': key.ss58_address, 'crypto_type': key.crypto_type}
        ticket_dict['ticket'] = ticket
        return ticket_dict

    ticket = create
    

    def ticket2address(self, ticket):
        """
        Get the address from a ticket
        """
        return ticket['address']
    
    
    def verify(self, data,  max_age:str=5,  age=None, timeout=None, **kwargs):
        max_age = age or timeout or max_age 
        staleness = c.time() - data['time']
        assert staleness < max_age, f"Staleness: {staleness} > {max_age}"
        ticket = data.get('ticket')
        address = ticket['address']
        signature = ticket.pop('signature')
        return c.verify(ticket, signature=signature, address=address, seperator=self.signature_seperator,  **kwargs)


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