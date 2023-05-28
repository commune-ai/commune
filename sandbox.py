import commune as c
# import bittensor as bt
import streamlit as st

# list keys


# add key



class Users(c.Module):
    def __init__(self):
        self.users  = []
        self.keys = st.write(c.keys())

    # @c.st_sidebar
    def sidebar(self):

        with st.form(key='my_form'):
            st.write("Add a key")
            key = st.text_input(label='Key')
            submit_button = st.form_submit_button(label='Submit')
        
    @classmethod
    def st(cls):
        self = Users()
        c.st_sidebar(self.sidebar)()
        
# Users.st()

c.Module().info()