
from typing import Any



def merge_dicts(a: Any, b: Any, 
                include_hidden:bool=False, 
                allow_conflicts:bool=True):
    '''
    Merge the dictionaries of a python object into the current object
    '''
    for b_k,b_v in b.__dict__.items():
        
        if include_hidden == False and (b_k.startswith('__') and b_k.endswith('__')):
            #i`f the function name starts with __ then it is hidden
            continue
        
        if not allow_conflicts:
            assert not hasattr(a, b_k), f'attribute {b_k} already exists in {a}'
        a.__dict__[b_k] = b_v
        
    return a
    
def merge_functions(a: Any, b: Any, 
                    include_hidden:bool=False, 
                    allow_conflicts:bool=True):
    '''
    Merge the functions of a python object into the current object
    '''
    for b_fn_name in dir(b):
        if include_hidden == False:
            #i`f the function name starts with __ then it is hidden
            if b_fn_name.startswith('__'):
                continue
            
        # if the function already exists in the object, raise an error
        if not allow_conflicts:
            assert not hasattr(a, b_fn_name), f'function {b_fn_name} already exists in {a}'
        
            
        # get the function from the python object
        try: 
            b_fn = getattr(b, b_fn_name)
        except NotImplementedError as e:
             print(e)
        if callable(b_fn):
            setattr(a, b_fn_name, b_fn)  
            
    return a
                        
def merge(a:Any, b: Any,
          include_hidden:bool=False, 
          allow_conflicts:bool=True) -> 'self':
    '''
    merge b into a and return a
    '''
    
    # merge the attributes
    a = merge_dicts(a=a,b=b, include_hidden=include_hidden, allow_conflicts=allow_conflicts)
    
    # merge the functions
    a = merge_functions(a=a,b=b, include_hidden=include_hidden, allow_conflicts=allow_conflicts)
    
    return a
    
    
        