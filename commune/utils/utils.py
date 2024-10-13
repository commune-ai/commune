
import commune as c
class Utils(c.Module):
    
    @classmethod
    def utils_paths(cls, search=None):
        utils = c.find_functions(c.root_path + '/utils')
        if search != None:
            utils = [u for u in utils if search in u]
        return sorted(utils)
    

    @classmethod
    def util2code(cls, search=None):
        utils = cls.utils()
        util2code = {}
        for f in utils:
            if search != None:
                if search in f:
                    util2code[f] = c.fn_code(f)
        return util2code