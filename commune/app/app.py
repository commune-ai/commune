import commune as c
import os
import json


class App(c.Module):

    def start(self,
           module:str = 'app', 
           fn='app', 
           port=8501, 
           public:bool = False, 
           remote:bool = False, 
           kwargs=None):
        kwargs = kwargs or {}
        if public:
            port = c.free_port()

        while c.port_used(port):
            c.print(f'Port {port} is already in use', color='red')
            port = port + 1
        if remote:
            remote_kwargs = c.locals2kwargs(locals())
            remote_kwargs['remote'] = False
            c.remote_fn(module=module, fn='st', kwargs=remote_kwargs)
            url = f'http://{c.ip()}:{port}'

            return {'success': True, 
                    'msg': f'running {module} on {port}', 
                    'url': url}
        module = c.module(module)
        module_filepath = module.filepath()
        # add port to the command
        cmd = f'streamlit run {module_filepath} --server.port {port}'
        
        if kwargs == None:
            kwargs = {}

        kwargs_str = json.dumps(kwargs)
        kwargs_str = kwargs_str.replace('"', "'")

        cmd += f' -- --fn {fn} --kwargs "{kwargs_str}"'

        module2dashboard = self.get('module2dashboard', {})
        if module in module2dashboard:
            try:
                module_port = module2dashboard[module]['port']
                c.kill_port(module_port)
            except Exception as e:
                c.print(f'Error: {e}', color='red')
        path = module.path()
        module2dashboard[path] = {
            'port': port,
            'fn': fn,
            'kwargs': kwargs,
            'cmd': cmd
        }
        c.print(module2dashboard)
        self.put('module2dashboard', module2dashboard)
        cwd = os.path.dirname(module_filepath)
        
        return c.cmd(cmd, verbose=True, cwd=cwd)
    


    def module2dashboard(self):
        return self.get('module2dashboard', {})
    

    def app(self):
        import streamlit as st
        st.write('Hello World!')
    

App.run(__name__)
    

