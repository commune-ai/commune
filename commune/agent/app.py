import commune as c
import streamlit as st

class App(c.Module):

    def app(self):
        st.write('hey')

App.run(__name__)