import commune as c
import os

class History(c.Module):
    def __init__(self, history_path='history', key=None):
        self.key = c.get_key(key)
        self.history_path = self.resolve_path(history_path)
        assert os.path.isdir(self.history_path), f"History path {self.history_path} does not exist"
        c.print(f"History path: {self.history_path}", color='green')
        c.print('set key:', self.key.ss58_address, color='green')

        
    def add_item(self, item:dict):
        assert isinstance(item, dict), f"Item must be a dictionary, got {type(item)}"        
        item['timestamp'] = item.get('timestamp', c.timestamp())
        path = self.key_history_path + '/' + str(item['timestamp'])
        return self.put(path, item)
    
    @property
    def key_history_path(self):
        return self.history_path + '/' + self.key.ss58_address
    
    def paths(self, key=None, max_age=None):
        files = []
        current_timestamp = c.timestamp()
        for file in c.ls(self.key_history_path):
            timestamp = self.get_file_timestamp(file)
            if max_age and current_timestamp - timestamp > max_age:
                continue
            files.append(file)
        return files
    
    def history(self, key=None, max_age=None):
        return [file for file in self.history(key, max_age)]
    
    def time2file(self):
        time2file = {}
        for path in self.paths():
            time2file[self.get_file_timestamp(path)] = path
        return time2file
        
                
            
    
    def get_file_timestamp(self, file):
        return int(file.split('/')[-1].split('.')[0])
    def get_file_age(self, file):
        return c.timestamp() - self.get_file_timestamp(file)
       
    @classmethod
    def test(cls):
        self = cls('test')
        self.add_history({'a': 'b', 'timestamp': c.timestamp()})
        c.print(self.history())


    

