import commune as c
import streamlit as st
import random

class KeyApp(c.Module):
    name2fn = {
        'Select Key': 'select_key',
        'Create Key': 'create_key', 
        'Rename Key': 'rename_key',
        'Remove Key': 'remove_key',
        'Ticket': 'ticket',
        'Verify Ticket': 'verify_ticket'
    }
    
    def __init__(self):
        self.sync()

    
    def sync(self):
        self._max_width_()
        self.local_css("style.css")
        # Load CSS
        @st.cache_resource()
        def load_keys():
            return c.keys()
        
        self.keys = load_keys()
        self.key2index = {k:i for i,k in enumerate(self.keys)}
        
    def local_css(self, file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
  
    def select_key(self):
        key = 'module'
        key = st.selectbox('Select Key', self.keys, index=self.key2index[key])
        self.key = c.get_key(key)
        if self.key.path is None:
            self.key.path = key
        st.write('Address: ', self.key.ss58_address)

    def create_key(self, expander=False):
        new_key = st.text_input('Name of Key', '', key='create')
        create_key_button = st.button('Create Key')
        if create_key_button and len(new_key) > 0:
            c.add_keys(new_key)
            key = c.get_key(new_key)

    def rename_key(self):
        old_key = st.selectbox('Select Key', self.keys, index=self.key2index[self.key.path], key='select old rename key')           
        new_key = st.text_input('New of Key', '', key='rename')
        rename_key_button = st.button('Rename Key')
        if rename_key_button and len(new_key) > 0:
            if c.key_exists(new_key):
                st.error('Key already exists')
            c.rename_key(old_key, new_key)
            key = c.get_key(new_key)
    
    def remove_key(self):
        rm_keys = st.multiselect('Select Key(s) to Remove', self.keys, [], key='rm_key')
        rm_key_button = st.button('Remove Key')
        if rm_key_button:
            c.rm_keys(rm_keys)

    def ticket_key(self):
        ticket_data = st.text_input('Ticket Data', '', key='ticket')
        ticket_button = st.button('Ticket Key')
        if ticket_button:
            ticket = self.key.ticket(ticket_data)
            st.write('Ticket')
            st.code(ticket)

    def app(self):
        st.write('# Key App')
        with st.expander('Description'):
            st.markdown(self.description)
        names = list(self.name2fn.keys())
        cols = st.columns(2)
        selected_fns = cols[0].multiselect('Select Options', names, ['Select Key'], key='select')
        num_cols = cols[1].number_input('Number of Columns', 0, 5, 1, 1)

        def app_wrapper(fn):
            try:
                with st.expander(fn, expanded=True):
                    getattr(self, self.name2fn[fn])()
            except Exception as e:
                try:
                    getattr(self, self.name2fn[fn])()
                except Exception as e:
                    st.error(e)


        if num_cols > 1:
            cols = st.columns(num_cols)
            for i, fn in enumerate(selected_fns):
                with cols[i % num_cols]:
                    app_wrapper(fn)

        else:
            for fn in selected_fns:
                
                with st.expander(fn, expanded=True):
                    app_wrapper(fn)

    def squares(self, cols=3):
        color_classes = ['color1', 'color2', 'color3', 'color4', 'color5', 'color6']
        cols = st.columns(cols)
        for i in range(9):
            color_class = random.choice(color_classes)
            with cols[i % cols]:
                st.markdown(f'<div class="square {color_class}">{i**2}</div>', unsafe_allow_html=True)

    def ticket(self, *args, **kwargs):
        data = st.text_input('Data', 'None')
        generate_ticket = st.button('Generate Ticket')
        if generate_ticket:
            ticket = self.key.ticket(data)
        else:
            ticket = None
        if ticket:
            st.write('Ticket')
            st.code(ticket)
        self.ticket = ticket
        self.verify_ticket()

    def verify_ticket(self):
        ticket = st.text_input('Enter the Ticket', self.ticket)
        st.write('FORMAT')
        st.code('data={data}time={time}::address={address}::signature={signature}')
        verify_ticket = st.button('Verify Ticket')
        if verify_ticket:
            
            st.write(ticket)
            result = c.verify_ticket(ticket)
            st.write('Result')

            st.write(result)
        else:
            result = None
        return result

    # css injection
    def _max_width_(self,         max_width_str = "max-width: 1900px;"):
        st.markdown(
            f"""
        <style>
        .block-container {{
            {max_width_str}
            }}
        .custom-widget {{
            display: grid;
            border: 1px solid black;
            padding: 12px;
            border-radius: 5%;
            color: #003366;
            margin-bottom: 5px;
            min-height: 251.56px;
            align-items: center;
        }}
        h6 {{
            display: block;
            font-size: 18px;
            margin-left: 0;
            margin-right: 0;
            font-weight: bold;
            color: #003366;
        }}
        h2 {{
            text-decoration: underline;
        }}
        h1 {{
            display: grid;
            justify-content: center;
            align-items: center;
        }}

        .css-1m8p54g{{
            justify-content: center;
        }}
        .css-1bt9eao {{
        }}
        .row-widget.stCheckbox {{
            display: grid;
            justify-content: center;
            align-items: center;
            border: solid 2px black;
            border-radius: 3%;
            height: 50px;
            background-color: #DF1B88;
            color: #FFFFFF;
        }}
        .css-1djdyxw {{
            color: #FFFFFF;
        }}
        .css-ps6290 {{
            color: black;
        }}
        .css-1cpxqw2 {{
            background-color: #00AB55;
            color: white;
            font-weight: 500;
            border: 1px solid #003366;
        }}
        <style>
        """,
            unsafe_allow_html=True,
        )

    description = '''

    # Key Management App

    This is a Key Management App built using `commune` and `streamlit`. The app provides functionalities to select, create, rename, and remove keys, as well as manage tickets and verify them.

    ## Select Key
    Select a key from the list of keys.
    ## Create Key
    Create a new key.
    ## Rename Key
    Rename an existing key.
    ## Remove Key
    Remove a key.
    ## Ticket
    Generate a ticket.
    ## Verify Ticket
    Verify a ticket.


    '''



b = KeyApp.run(__name__)


