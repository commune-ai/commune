
import commune as c
from typing import Dict , Any, List
import json
import os



def test_blacklisting():
    self = c.module('user')()
    key = c.get_key('test')
    assert key.ss58_address not in self.blacklist(), 'key already blacklisted'
    self.blacklist_user(key.ss58_address)
    assert key.ss58_address in self.blacklist(), 'key not blacklisted'
    self.whitelist_user(key.ss58_address)
    assert key.ss58_address not in self.blacklist(), 'key not whitelisted'
    return {'success': True, 'msg': 'blacklist test passed'}
    
