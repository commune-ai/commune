import commune
import streamlit as st
servers = commune.servers()


class Launchpad(commune.Module):

    def __init__(self, module: str = None):
        pass


    @property
    def servers(self):
        return commune.servers()
    
    def streamlit_sidebar(self):
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
        
    
Launchpad.streamlit()