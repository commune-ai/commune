import commune as c
import streamlit as st


class App(c.Module):

    def dashboard(cls):
        # disable the run_loop to avoid the background  thread from running
        self = c.module('vali')(start=False)
        c.load_style()
        module_path = self.path()
        c.new_event_loop()
        st.title(module_path)
        servers = c.servers(search='vali')
        server = st.selectbox('Select Vali', servers)
        state_path = f'dashboard/{server}'
        module = c.module(server)
        state = module.get(state_path, {})
        server = c.connect(server)
        if len(state) == 0 :
            state = {
                'run_info': server.run_info,
                'module_infos': server.module_infos(update=True)
            }

            self.put(state_path, state)

        module_infos = state['module_infos']
        df = []
        selected_columns = ['name', 'address', 'w', 'staleness']

        selected_columns = st.multiselect('Select columns', selected_columns, selected_columns)
        search = st.text_input('Search')

        for row in module_infos:
            if search != '' and search not in row['name']:
                continue
            row = {k: row.get(k, None) for k in selected_columns}
            df += [row]
        df = c.df(df)
        if len(df) == 0:
            st.write('No modules found')
        else:
            default_columns = ['w', 'staleness']
            sorted_columns = [c for c in default_columns if c in df.columns]
            df.sort_values(by=sorted_columns, ascending=False, inplace=True)
        st.write(df)
