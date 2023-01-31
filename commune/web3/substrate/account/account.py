from substrateinterface import SubstrateInterface, Keypair
from typing import List, Dict, Union
import commune


class SubstrateAccount(Keypair):
    def __init__(self, keypair:Union[Keypair, dict] = None, *args, **kwargs):
        if keypair:
            self.set_keypair(keypair)
        else:
            Keypair.__init__(self, *args, **kwargs)
        
        
    def set_keypair(self, keypair:Union[Keypair, dict]):
        if isinstance(keypair, dict):
            keypair = Keypair(**keypair)
        assert isinstance(keypair, Keypair), 'keypair must be a Keypair instance'
        self = commune.merge(self, keypair)
    @property
    def address(self):
        return self.ss58_address

    @classmethod
    def from_uri(cls, uri):
        """ Create a SubstrateAccount from a URI.
        """
        if not uri.startswith('//'):
            uri = '//' + uri
        
        keypair =  cls(keypair=cls.create_from_uri(uri))
        # keypair = cls.create_from_uri(uri)

        return keypair

    @classmethod
    def test_accounts(cls, demo_uris:List[str] = ['alice', 'bob', 'chris', 'billy', 'dave', 'sarah']) -> Dict[str, 'SubstrateAccount']:
        '''
        This method is used to create demo accounts for testing purposes.
        '''
        
        demo_accounts = {}
        for demo_uri in demo_uris:
            demo_accounts[demo_uri] =  cls.from_uri(demo_uri)
            
        
        return demo_accounts 
            
        

if __name__ == '__main__':
    module = SubstrateAccount()
    st.write(module)
