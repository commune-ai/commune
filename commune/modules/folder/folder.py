import commune as c

class Folder(c.Module):
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())

    def file2text(self, path, local=True):
        path = self.resolve_path(path)
        assert c.isdir(path), "Needs to be a dir"
        paths = c.glob(path)
        
        file2text  = {p:c.get_text(p) for p in paths}
        if local:
            if path[-1] != '/':
                path = path + '/'
            file2text = {k.replace(path, ''):v for k,v in file2text.items()}

        return file2text
    
    def cp(self, path:str, to:str):
        t1 = c.time()
        file2text = self.file2text(path)
        to = self.resolve_path(to)
        if not c.isdir(to):
            c.mkdir(to)
        for path,text in file2text.items():
            num_bytes = c.sizeof(text)
            new_path = to+'/'+path
            c.print(f'Saving {num_bytes} bytes -> {path}')
            c.put_text(new_path, text)

        latency = c.time() - t1 
        
        return {'to': to, 'latency': latency}
    
    
    