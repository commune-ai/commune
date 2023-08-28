import commune as c

class Password(c.Module):
    def __init__(self, **kwargs):
        config = self.set_config(config=kwargs)

    @classmethod
    def pwdmap(cls, search=None):
        pwds =  [f.split('/')[-1].split('.')[0]for f in c.ls(cls.tmp_dir())]
        if search:
            pwds = [p for p in pwds if search in p]
        return pwds




