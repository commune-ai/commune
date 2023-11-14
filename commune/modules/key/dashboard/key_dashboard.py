import commune as c
import streamlit as st

class KeyDashboard(c.Module):

    def __init__(self, state: dict=None):

        self.keys = c.keys()
        self.key2index = {k:i for i,k in enumerate(self.keys)}



       
    def select_key(self, expanded:bool = True):
        if expanded:
            with st.expander('Key', expanded=False):
                return self.select_key(expanded=False)

        key = 'module'
        key = st.selectbox('Select Key', self.keys, index=self.key2index[key])
        self.key =  c.get_key(key)
        if self.key.path == None:
            self.key.path = key


        st.write('Address: ', self.key.ss58_address)


    def create_key(self):
        with st.expander('Create Key', expanded=False):                
            new_key = st.text_input('Name of Key', '', key='create')
            create_key_button = st.button('Create Key')
            if create_key_button and len(new_key) > 0:
                c.add_key(new_key)
                key = c.get_key(new_key)

    def rename_key(self):
        with st.expander('Rename Key', expanded=False):    
            old_key = st.selectbox('Select Key', self.keys, index=self.key2index[self.key.path], key='select old rename key')           
            new_key = st.text_input('New of Key', '', key='rename')
            rename_key_button = st.button('Rename Key')
            replace_key = st.checkbox('Replace Key')
            if rename_key_button and len(new_key) > 0:
                if c.key_exists(new_key) and not replace_key:
                    st.error('Key already exists')
                c.rename_key(old_key,new_key)
                key = c.get_key(new_key)
    
    def remove_key(self):       
        with st.form(key='Remove Key'):            
            rm_keys = st.multiselect('Select Key(s) to Remove', self.keys, [], key='rm_key')
            rm_key_button = st.form_submit_button('Remove Key')
            if rm_key_button:
                c.rm_keys(rm_keys)

    @classmethod
    def dashboard(cls, *args, **kwargs):
        self = cls(*args, **kwargs)
        self.select_key()
        self.create_key()
        self.rename_key()
        self.remove_key()
        return self.key

KeyDashboard.run(__name__)

