import commune

class Launchpad(commune.Module):
    
    
    @classmethod
    def train(cls, models=[f'model::gptj::{i}' for i in [0]], 
              datasets=['dataset::bittensor']):
        
        model_module = commune.get_module('model.transformer')
        for model in models:
            for dataset in datasets:
                model_module.launch(fn='remote_train',name=f'train::{model}', kwargs={'model': model, 'dataset': dataset, 'save': True}, serve=False)
    
    
    @classmethod
    def deploy_fleet(cls, modules=None):
        modules = modules if modules else ['model.transformer', 'dataset.text.huggingface']
            
        print(modules)
        for module in modules:
            
            module_class = commune.get_module(module)
            assert hasattr(module_class,'deploy_fleet'), f'{module} does not have a deploy_fleet method'
            commune.get_module(module).deploy_fleet()

    
    @classmethod
    def deploy_models(cls):
        '''
        ArXiv/            Gutenberg_PG/
        BookCorpus2/      HackerNews/
        Books3/           NIHExPorter/
        DMMathematics/    OpenSubtitles/
        '''
        cls.deploy_fleet(module='model.transformer')

    @classmethod
    def deploy_datasets(cls):
        '''
        ArXiv/            Gutenberg_PG/
        BookCorpus2/      HackerNews/
        Books3/           NIHExPorter/
        DMMathematics/    OpenSubtitles/
        '''
        dataset_module = cls.deploy_fleet('dataset.text.huggingface')
        


    
    def streamlit_sidebar(self):
        import streamlit as st
        server = st.sidebar.selectbox("Select a server", self.servers)
        self.module = commune.connect(server)
        self.function_schema_map = self.module.function_schema_map()
        self.functions = list(self.function_schema_map.keys())
        # Define the custom CSS style
        custom_css = """
        <style>
            .selectbox label {
                font-size: 30px;
            }
            .selectbox select {
                font-size: 52px;
            }
        </style>
        """

        # Add the custom CSS to the Streamlit app
        st.markdown(custom_css, unsafe_allow_html=True)

        server = st.sidebar.selectbox("Select a Function", self.functions, 0)

        
        st.write(self.function_schema_map)
            
            
    @classmethod
    def streamlit_markdown(cls):
        import streamlit as st
        st.markdown(
        """
        <style>
        .reportview-container .markdown-text-container {
            font-family: monospace;
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            font-size: 20px;
            color: blue;
        }

        .sidebar>label {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            font-size: 20px;
            color: blue;
        }
        .Widget>label {
            color: white;
            font-family: monospace;
        }
        [class^="st-b"]  {
            color: white;
            font-family: monospace;
        }
        .st-bb {
            background-color: transparent;
        }
        .st-at {
            background-color: #0c0080;
        }
        footer {
            font-family: monospace;
        }
        .reportview-container .main footer, .reportview-container .main footer a {
            color: #0c0080;
        }
        header .decoration {
            background-image: none;
        }

        </style>
        """,
            unsafe_allow_html=True,
        )

    @classmethod
    def streamlit(cls):
        self = cls()
        self.streamlit_markdown()
        self.streamlit_sidebar()
        
    
        
if __name__ == "__main__":
    Launchpad.run()