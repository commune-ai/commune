
from typing import Any

def merge_dicts(a: Any, b: Any, include_hidden:bool=False):
    '''
    Merge the dictionaries of a python object into the current object
    '''
    for k,v in b.__dict__.items():
        if include_hidden == False:
            #i`f the function name starts with __ then it is hidden
            if k.startswith('__'):
                continue
        a.__dict__[k] = v
        
    return a
    
def merge_functions(a: Any, b: Any, include_hidden:bool=False):
    '''
    Merge the functions of a python object into the current object
    '''
    for a_fn_name in dir(a):
        if include_hidden == False:
            #i`f the function name starts with __ then it is hidden
            if a_fn_name.startswith('__'):
                continue
        # get the function from the python object
        a_fn = getattr(a, a_fn_name)
        if callable(fn):
            setattr(a, a_fn_name, a_fn)  
            
    return a
                        
def merge(a:Any, b: Any, include_hidden:bool=False) -> 'self':
    '''
    merge b into a and return a
    '''
    
    # merge the attributes
    merge_dict(a=a,b=b, include_hidden=include_hidden)
    
    # merge the functions
    merge_functions(a=a,b=b, include_hidden=include_hidden)
    
    return a
    
    
        