
import streamlit as st
import commune as c
from commune.api.api import ApiManager
class App(c.Module):
    def app(self):
    
        st.title("API Key Manager")

        # Create an instance of ApiManager
        api_manager = ApiManager()
        api_keys = api_manager.api_keys
        with st.expander("View API"):
            refresh = st.button("Refresh")
            st.write(api_keys)
        

        # Sidebar menu
        menu = ["Add API", "Remove API"]
        api_names =  list(api_keys.keys())
        choice = st.selectbox("Select an option", menu)

        if choice == "Add API":
            new_api_check = st.checkbox("New API")
            if new_api_check:
                name = st.text_input("Name")
            else:
                name = st.selectbox("Select API Name", api_names)
            api_key = st.text_input("API Key")
            if st.button("Add"):
                result = api_manager.add_api_key(name, api_key)
                st.success(result['msg'])
                st.info(f"Number of keys for {name}: {result['num_keys']}")
        elif choice == "Remove API":
            st.subheader("Remove API")
            name = st.selectbox("Select API Name", api_names)
            selected_rm_keys = st.multiselect("Select API Keys to remove", api_keys.get(name, []))
            if st.button("Pop"):
                try:
                    for key in selected_rm_keys:
                        st.success(api_manager.remove_api_key(name, key))
                except ValueError as e:
                    st.error(str(e))


 
        

if __name__ == '__main__':
    App.run()