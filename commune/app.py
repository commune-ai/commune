    

import commune as c
import streamlit as st

class App(c.Module):
    def __init__(self):
        c.new_event_loop()
        self.live_models = c.models()
        self.live_datasets = c.datasets()
        self.run()
        
    def selection(self):
        st.write('# Commune')
        with st.expander('Models', True):
            self.models = st.multiselect('select models',self.live_models, default=self.live_models)
            model_configs = c.call_pool(self.models, fn='config')
            self.model_configs = {k:v for k,v in zip(self.models, model_configs)}
            

    def run(self):
        self.selection()
        model_config = self.model_configs[model]
        stats = model_config.pop('stats')
        params = model_config
        info = params.pop('info')
        
        with st.expander('## Stats'):
            st.write(stats)
        
        with st.expander('## Params'):
            st.write(params)
            
            
        with st.expander('## Info'):
            st.write(info)
        st.write(info)
        
if __name__ == "__main__":
    
    App()



    @classmethod
    def dashboard(cls):
        
        st.write('hello world')
        st.write(c.models())