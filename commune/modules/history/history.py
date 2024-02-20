import commune as c

class History(c.Module):
    def __init__(self, key=None, history_path='history'):
        self.key = c.get_key(key)
        self.history_path = history_path
        c.print(f"History path: {self.history_path}", color='green')
        
    def add_history(self, item:dict,  key=None):
        path = self.history_path+'/' + self.key.ss58_address + '/' + str(item['timestamp'])
        return self.put(path, item)
    
    @classmethod
    def history(cls, key=None, history_path='history'):
        key = c.get_key(key)
        return cls.ls(history_path + '/' + key.ss58_address)
    
    
    @classmethod
    def all_history(cls, key=None, history_path='history'):
        key = c.get_key(key)
        return cls.glob(history_path)
        
    @classmethod
    def rm_key_history(cls, key=None, history_path='history'):
        key = c.get_key(key)
        return cls.rm(history_path + '/' + key.ss58_address)
    
    @classmethod
    def rm_history(cls, key=None, history_path='history'):
        key = c.get_key(key)
        return cls.rm(history_path)

