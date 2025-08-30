import commune as c

class Test:
    def __init__(self, module='api.test'):
        self.module = module
        self.api = c.module('apimanager')(module=module)
        
    def test_set_module(self):
        # Test with string
        self.api.set_module(self.module)
        assert self.api.module == self.module
        

    def test_add_key(self):
        test_key = 'test_key_123'
        result = self.api.add_key(test_key)
        assert test_key in result['keys']
        assert test_key in self.api.keys()
        
    def test_set_keys(self):
        test_keys = ['key1', 'key2', 'key3']
        self.api.set_keys(test_keys)
        stored_keys = self.api.keys()
        assert set(test_keys) == set(stored_keys)
    def test_rm_key(self):
        # First add a key
        test_key = 'key_to_remove'
        self.api.add_key(test_key)
        
        # Then remove it
        result = self.api.rm_key(self.api.module, test_key)
        assert test_key not in result['keys']
        assert test_key not in self.api.keys()

    def test_get_key(self):
        test_keys = ['key1', 'key2']
        self.api.set_keys(test_keys)
        key = self.api.get()
        assert key in test_keys

    def test_get_api_path(self):
        expected_path = c.resolve_path(self.api.path + '/' + self.api.module + '.json')
        assert self.api.get_api_path() == expected_path

    def test_apis(self):
        # Add some test APIs
        self.api.set_keys(['key1'], 'api1')
        self.api.set_keys(['key2'], 'api2')
        apis = self.api.apis()
        assert isinstance(apis, list)
        assert 'api1' in apis
        assert 'api2' in apis


    def test_rm_api(self):
        # First add an API
        test_module = 'test_api_remove'
        self.api.set_keys(['key1'], test_module)
        
        # Then remove it
        self.api.rm_api(test_module)
        assert test_module not in self.api.apis()

    def tearDown(self):
        # Clean up any test files/data
        self.api.rm_api(self.module)
        assert self.module not in self.api.apis()


    def test_fns(self):
        test_fns = []
        for fn in dir(self):
            if fn.startswith('test_'):
                test_fns.append(fn)
        return test_fns

    def run(self):
        results = {}
        for fn in self.test_fns():

            try:
                getattr(self, fn)()
                results[fn] = True
            except Exception as e:
                results[fn] = c.detailed_error(e)

        self.tearDown()
        return results

