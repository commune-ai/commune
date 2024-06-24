
import os
import commune as c
class Test(c.Module):
    def test_file(self, k='test_a', v=1):
        c.put(k,v)
        assert self.exists(k), f'file does not exist ({k})'
        self.encrypt_file(k)
        c.print(self.get_text(k))
        self.decrypt_file(k)
        new_v = self.get(k)
        assert new_v == v, f'new_v {new_v} != v {v}'
        self.rm(k)
        assert not self.exists(k)
        assert not os.path.exists(self.resolve_path(k))
        return {'success': True, 'msg': 'test_file passed'}
 
    def test_folder_module_detector(self,positives = ['module', 'vali', 'client']):
        for p in positives:
            assert self.is_folder_module(p) == True, f'{p} is a folder module'
        return {'success': True, 'msg': 'All folder modules detected', 'positives': positives}
    