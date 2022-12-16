





import os, sys
sys.path.append(os.environ['PWD'])
import gradio
from commune import Module
from commune.utils import *
import pandas as pd
import time




class ClientModule(Module):
    default_config_path =  'gradio.client.module'

    def bro(self, bro='bro', sup='fam',   fam='bro', output_example={'bro': 'fuck you jesus'}):
        return output_example 


    def get_simple_module_path(self, module_path):
        if isinstance(module_path, list):
            module_path_list = []
            for path in module_path:
                new_module_path = self.get_simple_module_path(module_path)
                module_path_list.append(new_module_path)
            return module_path_list
        elif isinstance(module_path, str): 
            return module_path.replace('.module.', '.')
        else:
            raise NotImplementedError

    def get_full_module_path(self,module_path):

        if isinstance(module_path, list):
            module_path_list = []
            for path in module_path:
                new_module_path = self.get_full_module_path(module_path)
                module_path_list.append(new_module_path)
            return module_path_list
        elif isinstance(module_path, str): 
            return '.'.join(*module_path[:-1],'module', module_path[-1])
        else:
            raise NotImplementedError
     
    def get_module_dir(self, module_path):
        if isinstance(module_path, list):
            module_path_list = []
            for path in module_path:
                new_module_path = self.get_full_module_path(module_path)
                module_path_list.append(new_module_path)
            return module_path_list
        elif isinstance(module_path, str): 
            simple_module_path = self.get_simple_module_path(module_path)
            module_dir = '.'.join(simple_module_path.split('.')[:-1])
            return module_dir
        else:
            raise NotImplementedError
     

    def get_module_object(self,module_path):
        if isinstance(module_path, list):
            module_path_list = []
            for path in module_path:
                new_module_path = self.get_module_object(module_path)
                module_path_list.append(new_module_path)
            return module_path_list
        elif isinstance(module_path, str): 
            return module_path.split('.')[-1]
        else:
            raise NotImplementedError

    @property
    def module_paths(self):
        return self.get_modules('full')

    @property
    def module_objects(self):
        return self.get_modules('object')

    @property
    def module_simple_paths(self):
        return self.get_modules('simple')

    @property
    def module_list(self):
        return self.client.rest.get(endpoint='list', params={'path_map':False})
    

    def get_modules(self, mode='simple', module_list=None, active=False):
        
        if active == True:
            port2module = self.port2module
            port_list = list(port2module.keys())
            module_list = list(port2module.values())

        else:
            module_list = self.module_list

            port_list = len(module_list)*['nan']

        modules_dict = dict(
            full  = module_list,
            simple = list(map(self.get_simple_module_path, module_list)), 
            object = list(map(self.get_module_object, module_list)),
            port = port_list,
            dir = list(map(self.get_module_dir, module_list))
            )

        st.write(modules_dict)
        return_type = None
        return_types = ['df','pandas']
        for k in return_types:
            trail_k = f'_{k}'
            if trail_k == mode[-len(trail_k):]:
                return_type = k
                mode =  mode[:-len(trail_k)]
                break

            
        

        
        if mode in  modules_dict: 
            if return_type in ['df','pandas']:  
                return pd.DataFrame({mode:modules_dict[mode]})
            else:
                return modules_dict[mode]

        elif '2' in mode:
            # {mode1}2{mode2} in terms of modules_dict
            modes = mode.split('2')
            assert len(modes) == 2, f'{modes} should be 2 but is {len(modes)}'
        
            if return_type in ['df','pandas']:  
                return pd.DataFrame({mode:modules_dict[mode] for mode in modes})
            else:
                return dict(zip(modules_dict[modes[0]], modules_dict[modes[1]]))

        else:
            raise NotImplementedError(mode)



    def port_active(self, port):
        port = str(port)
        active_ports = self.get_modules('port', active=True)
        return bool(port in active_ports)
    @property
    def num_active_modules(self):
        return len(self.get_modules('port', active=True))

    default_module = 'gradio.client.module.ClientModule'
    def add(self, **kwargs):
        default_kwargs = dict(module=self.default_module)
        kwargs = {**default_kwargs, **kwargs}
        
        prev_num_active_ports = deepcopy(self.num_active_modules)
        return_obj = self.client.rest.get(endpoint='add', params=kwargs)
        
        for i in range(10):
            time.sleep(0.5)
            changed_bool = self.num_active_modules  == prev_num_active_ports + 1
            
            if changed_bool:
                return return_obj

        raise Exception('Not Completed')

    
    def rm(self, **kwargs):

        default_kwargs = dict(module=self.default_module)
        kwargs = {**default_kwargs, **kwargs}
        return_object = self.client.rest.get(endpoint='rm', params=kwargs)
    
    @property
    def port2module(self):
        return self.client.rest.get(endpoint='port2module')

    @property
    def module2port(self):
        return self.client.rest.get(endpoint='module2port')

    @staticmethod
    def st_form(self, form='form', button='sync', fn=None, fn_kwargs={}, fn_args=[], submit_fn=None):
        
        with st.form(form):
            if fn != None:
                fn_output = fn(*fn_args, **fn_kwargs)

            submitted = st.form_submit_button(button)
            if submitted:
                if submit_fn != None:
                    submit_fn()
                
             
    

    def streamlit_tutorial(self):
        with st.expander('Module Tree (Map)', True):
            module_tree  = {}
            for k,v in self.get_modules('dir2object', active=False).items():
                dict_put(module_tree, k,v)
            st.write(module_tree)
        with st.expander(' Module2Paths', True):

            df = self.get_modules('dir2full_df', active=False)
            df['module'] = df['dir']
            df['folder_path'] = df['dir'].apply(lambda x: os.path.join(os.getenv('PWD'), x.replace('.', '/')))
            df = df.drop(columns=['dir', 'full'])
            st.write(df)


        port_module_df = self.get_modules('port2dir_df', active=True)
        if len(port_module_df) > 0:
            running_module_tags  = list(port_module_df.apply(lambda r: f"{r['dir']}:{r['port']}", axis=1))
        else:
            running_module_tags = []

        mode = 'add'
        with st.sidebar.expander(mode, True):
            with st.form(mode):
                simple2full_path = self.get_modules(mode='dir2full', active=False)
                simple_module_paths  = list(simple2full_path.keys())
                simple_module_selected = st.multiselect('Select a Module',simple_module_paths , ['gradio.client'])
                modules_selected = [simple2full_path[m] for m in simple_module_selected]
                submitted = st.form_submit_button(mode)
                if submitted:
                    for module_selected in modules_selected:
                        self.add(module=module_selected)

        port_module_df = self.get_modules('port2dir_df', active=True)







        with st.sidebar.expander('Active Modules', True):
            df = self.get_modules(mode='port2dir_df', active=True)

            df = df.rename(columns={'simple': 'path'})
            df['status'] = 'active'
            st.write(df)

            # module.client.rest.get(endpoint='add', params=dict(module='gradio.client.module.ClientModule'))
        

        running_module_tags  = list(df.apply(lambda r: f"{r['dir']}:{r['port']}", axis=1))
        mode = 'Running Modules'

        if len(df) > 0:
            running_module_tags  = list(df.apply(lambda r: f"{r['dir']}:{r['port']}", axis=1))
        else:
            running_module_tags = []

        with st.sidebar.expander(mode, True):

            running_ports = df['port']


            selected_running_module_tags = st.multiselect('',  running_module_tags, running_module_tags)
            
            selected_module_tags = [t for t in running_module_tags if t not in selected_running_module_tags]
            
            
            selected_ports = [t.split(':')[1] for t in selected_module_tags]

            for port in selected_ports:
                self.rm(port=port)

            remove_button = st.button('remove', selected_module_tags)
            if remove_button:
                selected_ports = [t.split(':')[1] for t in selected_running_module_tags]
                for port in selected_ports:
                    self.rm(port=port)


        sync_button = st.sidebar.button('Sync')
from commune.utils import *


if __name__ == "__main__":

    import streamlit as st
    module = ClientModule()
    module.streamlit_tutorial()
    

