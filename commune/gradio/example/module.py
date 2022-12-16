


import os, sys
sys.path.append(os.environ['PWD'])
import gradio as gr
from commune import Module
from inspect import getfile
import streamlit as st
import inspect
import socket
from commune.utils import SimpleNamespace

class ExampleModule(Module):
    default_config_path =  'gradio.example'
    def bro(self, input1=1, inpupt2=10, output_example={'bro': 1}):
        pass

    @classmethod
    def gradio(cls):
        return gr.Interface(lambda inputs : f'Hello {inputs}, welcome to the gradio example', inputs="text", outputs='text')        

    @classmethod
    def streamlit(cls):
        st.write("Hello World")
        st.write("")

if __name__ == "__main__":
    ExampleModule().run()

