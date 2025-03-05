import sys
import time
import sys
import commune as c
from typing import Any
print = c.print
class Cli:
    desc = 'cli for running functions'
    safety = True
    ai_enabled = False

    def get_params(self, argv):
        args = []
        kwargs = {}
        parsing_kwargs = False
        for arg in argv:
            if '=' in arg:
                parsing_kwargs = True
                key, value = arg.split('=')
                kwargs[key] = c.str2python(value)
            else:
                assert parsing_kwargs is False, 'Cannot mix positional and keyword arguments'
                args.append(c.str2python(arg))
        return {'args': args, 'kwargs': kwargs}

    def get_fn(self, 
                argv:list, 
                default_fn:str='forward',
                splitter='/', 
                default_module:str='module', 
                helper_fns=['code']):
        if len(argv) == 0:
            fn = default_fn
        else:
            fn = argv.pop(0).replace('-', '_')
        if fn.endswith('/'):
            fn += 'forward'
        # get the function object
        if splitter in fn:
            module = splitter.join(fn.split(splitter)[:-1])
            module = module.replace(splitter, '.')
            fn = fn.split(splitter)[-1]
        else:
            module = default_module
        if module.endswith('.py'):
            module = module[:-3]
        if module in c.shortcuts:
            old_module = module
            module = c.shortcuts[module]
            print(f'NameShortcut({old_module} -> {module})', color='yellow')
        filepath = c.filepath(module).replace(c.home_path, '~')    
        print(f'Call({module}/{fn}, path={filepath})', color='yellow')
        module = c.module(module)()
        return getattr(module, fn)

    def forward(self):
        if self.safety:
            assert c.pwd() != c.home_path, 'Cannot run cli in home directory, please change if you want to run'
        t0 = time.time()
        argv = sys.argv[1:]
        if len(argv) == 0:
            argv = ['vs']
        fn_obj = self.get_fn(argv) # get the function object
        params = self.get_params(argv) # get the parameters
        try:
            output = fn_obj(*params['args'], **params['kwargs']) if callable(fn_obj) else fn_obj
        except Exception as e:
            output = c.detailed_error(e)
        if self.ai_enabled:
            output = c.ask(output)
        latency = time.time() - t0
        print(f'❌Error({latency:.3f}sec)❌' if c.is_error(output) else f'✅Result({latency:.3f}s)✅')
        is_generator = c.is_generator(output)
        if is_generator:
            for item in output:
                if isinstance(item, dict):
                    print(item)
                else:
                    print(item, end='')
        else:
            print(output)
