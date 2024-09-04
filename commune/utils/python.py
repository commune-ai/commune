

    
from typing import *

def pip_libs(cls):
    return list(cls.lib2version().values())

required_libs = []

def ensure_libs(libs: List[str] = None, verbose:bool=False):
    results = []
    for lib in libs:
        results.append(ensure_lib(lib, verbose=verbose))
    return results

def install(cls, libs: List[str] = None, verbose:bool=False):
    return cls.ensure_libs(libs, verbose=verbose)

def ensure_env(cls):
    cls.ensure_libs(cls.libs)

def pip_exists(cls, lib:str, verbose:str=True):
    return bool(lib in cls.pip_libs())

def version(cls, lib:str=None):
    import commune as c
    lib = lib or c.libname
    lines = [l for l in cls.cmd(f'pip3 list', verbose=False).split('\n') if l.startswith(lib)]
    if len(lines)>0:
        return lines[0].split(' ')[-1].strip()
    else:
        return f'No Library Found {lib}'
    

def pip_exists(lib:str):
    return bool(lib in pip_libs())

def ensure_lib( lib:str, verbose:bool=False):
    if  pip_exists(lib):
        return {'lib':lib, 'version':version(lib), 'status':'exists'}
    elif pip_exists(lib) == False:
        pip_install(lib, verbose=verbose)
    return {'lib':lib, 'version':version(lib), 'status':'installed'}

def pip_install(lib:str= None,
                upgrade:bool=True ,
                verbose:str=True,
                ):
    import commune as c
    if lib in c.modules():
        c.print(f'Installing {lib} Module from local directory')
        lib = c.resolve_object(lib).dirpath()
    if lib == None:
        lib = c.libpath

    if c.exists(lib):
        cmd = f'pip install -e'
    else:
        cmd = f'pip install'
        if upgrade:
            cmd += ' --upgrade'
    return c.cmd(cmd, verbose=verbose)


# JUPYTER NOTEBOOKS
def enable_jupyter():
    import commune as c
    c.nest_asyncio()

jupyter = enable_jupyter

def pip_list(lib=None):
    import commune as c
    lib = lib or c.libname
    pip_list =  c.cmd(f'pip list', verbose=False, bash=True).split('\n')
    if lib != None:
        pip_list = [l for l in pip_list if l.startswith(lib)]
    return pip_list