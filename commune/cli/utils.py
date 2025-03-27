

def str2python(x):
    x = str(x)
    if isinstance(x, str) :
        if x.startswith('py(') and x.endswith(')'):
            try:
                return eval(x[3:-1])
            except:
                return x
    if x.lower() in ['null'] or x == 'None':  # convert 'null' or 'None' to None
        return None 
    elif x.lower() in ['true', 'false']: # convert 'true' or 'false' to bool
        return bool(x.lower() == 'true')
    elif x.startswith('[') and x.endswith(']'): # this is a list
        try:
            list_items = x[1:-1].split(',')
            # try to convert each item to its actual type
            x =  [str2python(item.strip()) for item in list_items]
            if len(x) == 1 and x[0] == '':
                x = []
            return x
        except:
            # if conversion fails, return as string
            return x
    elif x.startswith('{') and x.endswith('}'):
        # this is a dictionary
        if len(x) == 2:
            return {}
        try:
            dict_items = x[1:-1].split(',')
            # try to convert each item to a key-value pair
            return {key.strip(): str2python(value.strip()) for key, value in [item.split(':', 1) for item in dict_items]}
        except:
            # if conversion fails, return as string
            return x
    else:
        # try to convert to int or float, otherwise return as string
        
        for type_fn in [int, float]:
            try:
                return type_fn(x)
            except ValueError:
                pass
    return x