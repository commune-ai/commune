import commune as c
class Model(c.Module):
    def __init__(self,**kwargs):
        self.set_model(**kwargs)
    def set_model(self, **kwargs):
        from .models.imagebind_model import ImageBindModel
        self.merge_module(ImageBindModel(**kwargs))
    def merge_module(self, module):
        for k in dir(module):
            if not k.startswith('_'):
                setattr(self,k,getattr(module,k))

    @classmethod
    def install(cls, **kwargs):
        path =  cls.dirpath() + '/requirements.txt'
        c.cmd(f'pip3 install -r {path}')