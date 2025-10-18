import commune as c 
class Degit:
    """
    A base class that provides fundamental functionality for commune modules.
    """
    def __init__(self, path='~/.degit'):
        """
        Initialize the base class with configurable parameters.
        Args:
            **kwargs: Arbitrary keyword arguments to configure the instance
        """
        # Store configuration as a Munch object for dot notation access
        self.store = c.mod('store')(path)


    def put(self, mod='degit'):
        prev_cid = self.prev_cid(mod)
        cid = c.cid(mod)
        if prev_cid != cid: 
            content = c.content(mod)
            data  = {
                'content': content,
                'cid': cid,
                'prev_cid': prev_cid
                'time': c.time()
             }
            current_path = mod +'/current.json'
            if self.store.exists(current_path): 
                prev_cid_path = mod + f'/history/{prev_cid}'
                self.store.mv(current_path, prev_cid_path)
            self.store.put(path, data)
        else: 
            print('no need ot add new cid')
        return {''}



    def prev_cid(self, mod='degit'):
        path2age = self.store.path2age(mod)



        
