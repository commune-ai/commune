from typing import *

def find_lines(text:str, search:str) -> List[str]:
    """
    Finds the lines in text with search
    """
    found_lines = []
    lines = text.split('\n')
    for line in lines:
        if search in line:
            found_lines += [line]
    return found_lines

def find_code_lines( search:str = None , module=None) -> List[str]:
    import commune as c
    code = c.module(module).code()
    return find_lines(search=search, text=code)
