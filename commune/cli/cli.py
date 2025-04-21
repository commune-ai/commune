import sys
import time
import sys
from typing import Any
import inspect
import commune as c

import os
print = c.print
class Cli:

    def forward(self,
                fn='vs',  
                module='module', 
                default_fn = 'forward'):

        # ensure your not in the system home
        assert not os.path.abspath(os.getcwd()) == c.home_path, 'You are in the system home directory'
        t0 = time.time()
        argv = sys.argv[1:]
        # ---- FUNCTION
        if len(argv) == 0:
            argv += [fn]

        fn = argv.pop(0)
        local_modules = c.local_modules()
        if '/' in fn:
            if fn.startswith('/'):
                fn = fn[1:]
            if fn.endswith('/'):
                fn = fn + default_fn
            module =  c.module( '/'.join(fn.split('/')[:-1]).replace('/', '.'))()
            fn = fn.split('/')[-1]
            fn_obj = getattr(module, fn)
        else:
            module = c.module(module)()
            fn_obj = getattr(module, fn)
            
        params = {'args': [], 'kwargs': {}} 
        parsing_kwargs = False
        if len(argv) > 0:
            for arg in argv:
                if '=' in arg:
                    parsing_kwargs = True
                    key, value = arg.split('=')
                    params['kwargs'][key] = c.str2python(value)
                else:
                    assert parsing_kwargs is False, 'Cannot mix positional and keyword arguments'
                    params['args'].append(c.str2python(arg))        
        # run thefunction
        result = fn_obj(*params['args'], **params['kwargs']) if callable(fn_obj) else fn_obj
        speed = time.time() - t0
        module_name = module.__class__.__name__
        c.print(f'Call({module_name}/{fn}, speed={speed:.2f}s)')
        if c.is_generator(result):
            for item in result:
                c.print(item, end='')
        else:
            c.print(result)
