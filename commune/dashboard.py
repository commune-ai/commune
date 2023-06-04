import commune
import streamlit as st 
from typing import List, Dict, Union, Any 


class Dashboard(commune.Module):

    def __init__(self):
        self.public_ip = commune.external_ip()
        self.load_state()

    
    def load_state(self):
        self.namespace = commune.namespace()
        self.servers = list(self.namespace.keys())
        self.module_tree = commune.module_tree()
        self.module_list = ['module'] + list(self.module_tree.keys())
        sorted(self.module_list)
        
    


    def streamlit_module_browser(self):

        module_list =  self.module_list
        
   
        
        
        st.write(f'## {self.module_name}')
        
        self.streamlit_launcher()
        
        
        with st.expander(f'Module Function Info', False):
            st.write(self.module.schema)
   

            

    
    def streamlit_sidebar(self):
        with st.sidebar:  
                      
            self.module_name = st.selectbox('Module List',self.module_list, 0)   
            self.module = commune.module(self.module_name)

                
            self.streamlit_peers()

            with st.expander('Module Tree'):
                st.write(self.module_list)
            with st.expander('Servers'):
                st.write(self.servers)
            with st.expander('Peers'):
                st.write(self.peers())
            
            self.update_button = st.button('Update', False)

            if self.update_button:
                self.load_state()
                
            
    def streamlit_peers(self):

        

        peer = st.text_input('Add a Peer', '0.0.0.0:9401')
        cols = st.columns(3)
        add_peer_button = cols[0].button('add Peer')
        rm_peer_button = cols[2].button('rm peer')
        if add_peer_button :
            self.add_peer(peer)
        if rm_peer_button :
            self.rm_peer(peer)
            
            
        

        
 
    def streamlit_server_info(self):
        
        
        for peer_name, peer_info in self.namespace.items():
            with st.expander(peer_name, True):
                peer_info['address']=  f'{peer_info["ip"]}:{peer_info["port"]}'
                st.write(peer_info)
                
            
            
        
        
     

        peer_info_map = {}
 
    def function_call(self, module:str, fn_name: str = '__init__' ):
        
        module = commune.connect(module)
        peer_info = module.peer_info()
        
        function_info_map = self.module.function_info_map()
        fn_name = fn_name
        fn_info = function_info_map[fn_name]
        
        kwargs = {}
        cols = st.columns([3,1,6])
        cols[0].write(f'#### {name}.{fn_name}')
        cols[2].write('#### Module Arguments')
        cols = st.columns([2,1,4,4])
        launch_col = cols[0]
        kwargs_cols = cols[2:]
        st.write(function_info_map)
        with launch_col:
       
            name = st.text_input('**Name**', self.module_name) 
            tag = None
            # tag = st.text_input('**Tag**', 'None')
            if tag == 'None' or tag == '' or tag == 'null':
                tag = None
            refresh = st.checkbox('**Refresh**', False)
            # mode = st.selectbox('**Select Mode**', ['pm2',  'ray', 'local'] ) 
            mode = 'pm2'
            serve = True
            launch_button = st.button('Launch Module')  
            
        
        # kwargs_cols[0].write('## Module Arguments')
        for i, (k,v) in enumerate(fn_info['default'].items()):
            
            optional = fn_info['default'][k] != 'NA'
            fn_key = k 
            if k in fn_info['schema']:
                k_type = fn_info['schema'][k]
                if k_type.startswith('typing'):
                    k_type = k_type.split('.')[-1]
                fn_key = f'**{k} ({k_type}){"" if optional else "(REQUIRED)"}**'
            kwargs_col_idx  = i 
            if k in ['kwargs', 'args'] and v == 'NA':
                continue
            
            
            
            kwargs_col_idx = kwargs_col_idx % (len(kwargs_cols))
            init_kwarg[k] = kwargs_cols[kwargs_col_idx].text_input(fn_key, v)
            
        init_kwarg.pop('self', None )
        init_kwarg.pop('cls', None)
        if launch_button:
            kwargs = {}
            for k,v in init_kwarg.items():
                if v == 'None':
                    v = None
                elif k in fn_info['schema'] and fn_info['schema'][k] == 'str':
                    v = v
                elif k == 'kwargs':
                    continue
                elif v == 'NA':
                    assert k != 'NA', f'Key {k} not in default'
                else:
                    v = eval(v) 
                
                kwargs[k] = v
                
                
            launch_kwargs = dict(
                module = self.module,
                name = name,
                tag = tag,
                mode = mode,
                refresh = refresh,
                serve = serve,
                kwargs = kwargs,
            )
            st.write(launch_kwargs)
            commune.launch(**launch_kwargs)
            
    
    def streamlit_launcher(self):
        # function_map =self.module_info['funciton_schema_map'] = self.module_info['object'].get_function_schema_map()
        # function_signature = self.module_info['function_signature_map'] = self.module_info['object'].get_function_signature_map()
        module_schema = self.module.schema()
        st.write(module_schema)
        fn_name = '__init__'
        fn_info = module_schema[fn_name]
        
        init_kwarg = {}
        cols = st.columns([3,1,6])
        cols[0].write('#### Launcher')
        cols[2].write('#### Module Arguments')
        cols = st.columns([2,1,4,4])
        launch_col = cols[0]
        kwargs_cols = cols[2:]
        
        with launch_col:
       
            name = st.text_input('**Name**', self.module_name) 
            tag = None
            # tag = st.text_input('**Tag**', 'None')
            if tag == 'None' or tag == '' or tag == 'null':
                tag = None
            refresh = st.checkbox('**Refresh**', False)
            # mode = st.selectbox('**Select Mode**', ['pm2',  'ray', 'local'] ) 
            mode = 'pm2'
            serve = True
            launch_button = st.button('Launch Module')  
            
        st.write(fn_info)
            
        fn_info['default'].pop('self', None)
        fn_info['default'].pop('cls', None)
        
            
        # kwargs_cols[0].write('## Module Arguments')
        for i, (k,v) in enumerate(fn_info['default'].items()):
            
            optional = fn_info['default'][k] != 'NA'
            fn_key = k 
            if k in fn_info['schema']:
                k_type = fn_info['schema'][k]
                if k_type.startswith('typing'):
                    k_type = k_type.split('.')[-1]
                fn_key = f'**{k} ({k_type}){"" if optional else "(REQUIRED)"}**'
            kwargs_col_idx  = i 
            if k in ['kwargs', 'args'] and v == 'NA':
                continue
            
            
            
            kwargs_col_idx = kwargs_col_idx % (len(kwargs_cols))
            init_kwarg[k] = kwargs_cols[kwargs_col_idx].text_input(fn_key, v)
            
        init_kwarg.pop('self', None )
        init_kwarg.pop('cls', None)
        if launch_button:
            kwargs = {}
            for k,v in init_kwarg.items():
                if v == 'None':
                    v = None
                elif k in fn_info['schema'] and fn_info['schema'][k] == 'str':
                    v = v
                elif k == 'kwargs':
                    continue
                elif v == 'NA':
                    assert k != 'NA', f'Key {k} not in default'
                else:
                    v = eval(v) 
                
                kwargs[k] = v
                
                
            launch_kwargs = dict(
                module = self.module,
                name = name,
                tag = tag,
                mode = mode,
                refresh = refresh,
                serve = serve,
                kwargs = kwargs,
            )
            st.write(launch_kwargs)
            commune.launch(**launch_kwargs)
        
    def streamlit_playground(self):
        class bro:
            def __init__(self, a, b):
                self.a = a
                self.b = b
                
        st.write(str(type(commune)) == "<class 'module'>")
        st.write()
        pass
  
            
    def streamlit_resource_browser(self):

        import streamlit as st
        import plotly.graph_objects as go

        gpu_info = {
        "0": {
            "name": "NVIDIA RTX A6000",
            "free": 23.122280448,
            "total": 51.041271808,
            "used": 27.91899136
        },
        "1": {
            "name": "NVIDIA RTX A6000",
            "free": 7.370571776,
            "total": 51.041271808,
            "used": 43.670700032
        },
        "2": {
            "name": "NVIDIA RTX A6000",
            "free": 7.6054528,
            "total": 51.041271808,
            "used": 43.435819008
        },
        "3": {
            "name": "NVIDIA RTX A6000",
            "free": 7.6054528,
            "total": 51.041271808,
            "used": 43.435819008
        },
        "4": {
            "name": "NVIDIA RTX A6000",
            "free": 7.6054528,
            "total": 51.041271808,
            "used": 43.435819008
        },
        "5": {
            "name": "NVIDIA RTX A6000",
            "free": 7.6054528,
            "total": 51.041271808,
            "used": 43.435819008
        },
        "6": {
            "name": "NVIDIA RTX A6000",
            "free": 7.6054528,
            "total": 51.041271808,
            "used": 43.435819008
        },
        "7": {
            "name": "NVIDIA RTX A6000",
            "free": 21.129986048,
            "total": 51.041271808,
            "used": 29.91128576
        }
        }

        # Initialize empty lists for the data
        gpu_names = []
        free_memory = []
        used_memory = []

        # Loop through the GPU data and append the values to the lists
        for gpu_id, gpu_data in gpu_info.items():
            gpu_names.append(f"GPU {gpu_id}: {gpu_data['name']}")
            free_memory.append(gpu_data['free'])
            used_memory.append(gpu_data['used'])

        # Create the Plotly figure with multiple bar charts
        fig = go.Figure()
        fig.add_trace(go.Bar(x=gpu_names, y=free_memory, name="Free Memory", marker_color="green"))
        fig.add_trace(go.Bar(x=gpu_names, y=used_memory, name="Used Memory", marker_color="red"))

        # Set the chart title and axis labels
        fig.update_layout(title="GPU Memory Usage", xaxis_title="GPU", yaxis_title="Memory (GB)")

        # Display the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # with st.expander('GPUs'):
        #     gpu_info = commune.gpu_map()
            
        #     import plotly.graph_objects as go
        #     #culate the percentage of used and free memory for each GPU
        #     for gpu_id, gpu_data in gpu_info.items():
        #         gpu_data["used_pct"] = gpu_data["used"] / gpu_data["total"] * 100
        #         gpu_data["free_pct"] = gpu_data["free"] / gpu_data["total"] * 100

        #     # Calculate the number of rows and columns needed to tile the pie charts
        #     num_gpus = len(gpu_info)
        #     num_cols = min(num_gpus, 1)
        #     num_rows = (num_gpus - 1) // num_cols + 1

        #     # Create a grid of pie charts
        #     grid = []

        #     for i in range(num_rows):
        #         row = []
        #         cols = st.columns(num_cols)

        #         for j in range(num_cols):
        #             gpu_index = i * num_cols + j
        #             if gpu_index < num_gpus:
        #                 gpu_id = list(gpu_info.keys())[gpu_index]
        #                 gpu_data = gpu_info[gpu_id]

        #                 # Create a Plotly pie chart for the current GPU
        #                 fig = go.Figure(go.Pie(
        #                     values=[gpu_data["free_pct"], gpu_data["used_pct"]],
        #                     labels=["Free", "Used"],
        #                     marker_colors=["green", "red"],
        #                     textinfo="label+percent",
        #                 ))

        #                 # Set the chart title
        #                 # make it compact
        #                 fig.update_layout(showlegend=False)
        #                 cols[j].plotly_chart(fig)     
        #                 # Add the chart to
        
    @classmethod
    def streamlit(cls):
        commune.new_event_loop()

        commune.nest_asyncio()
        self = cls()
        st.set_page_config(layout="wide")
        
        self.streamlit_sidebar()
        tabs = st.tabs(['Module Launcher', 'Peers', 'Playground', 'Resources'])

        self.streamlit_module_browser()


        
        

if __name__ == '__main__':
    Dashboard.streamlit()