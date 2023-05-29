

import commune as c

class TestModule(c.Module):

    @classmethod
    def test_call_verification(cls):
        auth = c.call_auth_data(module='module', fn='test_call_verification', args=[], kwargs={})
        key = c.get_key('bro')
        auth = key.sign(auth)
        assert c.verify(auth)
        c.print(c.get_signer(auth))
        c.print(auth)
        
        
        
        
TestModule.test()      
