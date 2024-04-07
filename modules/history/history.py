import commune as c
import os

class History(c.Module):
    def __init__(self, folder_path='history'):
        self.folder_path = self.resolve_path(folder_path)



    def set_folder_path(self, path):
        self.folder_path = self.resolve_path(path)
        assert os.path.isdir(self.folder_path), f"History path {self.folder_path} does not exist"
        c.print(f"History path: {self.folder_path}", color='green')
        
    def add(self, item:dict, path=None):
        if 'timestamp' not in item:
            item['timestamp'] = c.timestamp()
        path = path or (self.folder_path + '/' + str(item['timestamp']))
        return self.put(path, item)
    
    def paths(self, key=None, max_age=None):
        files = []
        current_timestamp = c.timestamp()
        for file in c.ls(self.folder_path):
            timestamp = self.get_file_timestamp(file)
            if max_age and current_timestamp - timestamp > max_age:
                continue
            files.append(file)
        return files
    
    def get_file_timestamp(self, file):
        return int(file.split('/')[-1].split('.')[0])

    def history_paths(self, n=10, reverse=False):
        paths =  self.ls(self.folder_path)
        sorted_paths = sorted(paths, reverse=reverse)
        return sorted_paths[:n]
    
    def history(self, n=5, reverse=True, idx=None):
        history_paths = self.history_paths(n=n, reverse=reverse)
        history = [c.get(s) for s in history_paths]
        if idx:
            return history[idx]
        return history
    
    def last_n(self, n=1):
        return self.history(n=n)
    


    
