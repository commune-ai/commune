import commune as c
import streamlit as st
import os

class App(c.Module):

    def title(self):
        # Change the title of the app to 'Cyberbunk City'
        st.markdown('# Cyberbunk City')

    def app(self):
        self.title()
        self.agent = c.module('agent')()

        # Define the CSS for different buttons with 'cyberbunk' vibes
        st.markdown("""
            <style>
            .send-button>button {
                background-color: #8B00FF;
                color: white;
                border: 1px solid #8B00FF;
                width: 100%;
            }
            .stop-button>button {
                background-color: #00FFFF;
                color: black;
                border: 1px solid #00FFFF;
                width: 100%;
            }
            </style>
            """, unsafe_allow_html=True)


        resolve_path = lambda p: os.path.abspath(os.path.expanduser(p))
        code = None


        og_code_col, model_code_col = st.columns(2)

        cols = st.columns([2,5])
        folder_path = './'
        folder_path = cols[0].text_input('Folder Path', resolve_path(folder_path))
        folder_path = resolve_path(folder_path)
        python_files = [f for f in c.glob(folder_path) if f.endswith('.py')]
        num_files = len(python_files)
        filepath = cols[1].selectbox(f'Select File (n={num_files})', python_files)
        with st.expander(f'Code'):
            code = c.get_text(filepath)
            code = st.text_area('Code', code, height=400)

        input = st.text_area('Input')

        # Use columns to span the page
        col1, col2 = st.columns(2)

        send_button = st.button('Transmit', key='send', use_container_width=True)
        st.markdown('<div class="send-button"></div>', unsafe_allow_html=True)


        if send_button:
            kwargs = {'text': input, 'code': code, 'file': filepath}
            tx_id = c.hash(kwargs)
            st.write('Transaction ID:', tx_id)
            history_path = self.resolve_path(f'history/{self.key.ss58_address}')


            content = self.get(history_path, {})
            if 'data' not in content:
                response = self.agent.forward(**kwargs)
                def generator(response):
                    response = self.agent.forward(input, code=code,  stream=1)

                    content['data'] = ''
                    for r in response:
                        content['data'] += r
                        yield r
                st.write_stream(generator(response))
                self.put(history_path, content) 

            response = content['data']
                    
          
            with st.expander('Save Response'):
             
                response = response.split('```python')[-1].split('```').pop(0)
                st.write(response)

                save_filepath = st.text_input('Save File Path', filepath)
                save_button = st.button('Save', key='save', use_container_width=True)
            
                if save_button:
                    c.put_text(save_filepath, code)
                    st.write('Saved to', filepath)

    def process_response(self, code):
        return code.split('```python')[-1].split('```').pop(0)


App.run(__name__)