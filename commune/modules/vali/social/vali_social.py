import commune as c
import streamlit as st
import plotly.express as px

class ValiSocial(c.Module):
    def __init__(self, **kwargs):
        self.set_config(kwargs=kwargs)

    @classmethod
    def dashboard(cls):
        self = cls()
        c.load_style()
        st.write("## Social Vali")
       
        subspace = c.module('subspace')()


        features = ['key',  
                'address', 
                'name', 
                'emission', 
                'incentive', 
                'dividends', 
                'last_update', 
                'delegation_fee',
                'trust', 
                'regblock']
        
        default = [ 'name', 'emission', 'incentive', 'dividends', 'last_update', 'key',  'delegation_fee', 'regblock', 'trust', ]
        features = st.multiselect("Select features", features, default)
        update = st.button("Update")
        if update:
            self.update()

        modules = c.submit(subspace.modules, kwargs=dict(features=features), timeout=20)
        modules = c.wait(modules)
        lag = c.lag()
        st.write(f"Lag: {lag}")
        df = c.df(modules)
        st.write(df)
        self.histogram(df)


    def histogram(self, df, expander=True):
        
        if expander:
            with st.expander('Histogram'):
                self.histogram(df, False)
            return
        features = list(df.columns)
        histogram = st.checkbox('Histogram', False)
        idx2name = {i:name for i,name in enumerate(features)}
        name2idx = {name:i for i,name in enumerate(features)}
        histogram_key = st.selectbox('Select Histogram Key', features, name2idx['emission'])
        fig = px.histogram(df, x=histogram_key)
        st.plotly_chart(fig)

        # self.plot_dashboard(df)
        

    def update(self):
        if not c.server_exists('subspace'):
            c.submit(c.sync, kwargs=dict(module='subspace'), timeout=100)
            return


    @classmethod
    def plot_dashboard(cls, df):
        import plotly.express as px
        
        cols = list(df.columns)
        # bar_chart based on x and y

        if len(df) == 0:
            st.error('You are not staked to any modules')
            return 
        col2idx = {c:i for i,c in enumerate(cols)}
        defult_x_col = col2idx['name']
        default_y_col = col2idx['emission']

        plot_kwargs = {}

        st_cols = st.columns([1,3])

        with st_cols[0]:
            plot_type = st.selectbox('Select Plot Type', ['pie', 'bar', 'line', 'scatter', 'histogram', 'treemap'], 0, key='info.plot')

            if plot_type in [ 'bar', 'line', 'scatter']:
                plot_kwargs['x'] = st.selectbox('Select X', cols, defult_x_col)
                plot_kwargs['y'] = st.selectbox('Select Y', cols, default_y_col)
            elif plot_type in ['histogram']:
                plot_kwargs['x'] = st.selectbox('Select Value', cols, defult_x_col)
            elif plot_type in ['pie']:
                plot_kwargs['names'] = st.selectbox('Select Names', cols, defult_x_col)
                plot_kwargs['values'] = st.selectbox('Select Values', cols, default_y_col)
            elif plot_type in ['treemap']:
                plot_kwargs['path'] = st.multiselect('Select Path', cols, ["name"])
                plot_kwargs['values'] = st.selectbox('Select Values', cols, default_y_col)


            sort_type = st.selectbox('Sort Type', cols , 0)

            if sort_type in cols:
                ascending = st.checkbox('Ascending', False)
                df = df.sort_values(sort_type, ascending=ascending)

        with st_cols[1]:
            plot_fn = getattr(px, plot_type)
            plot_kwargs_title =  " ".join([f"{k.lower()}:{v}" for k,v in plot_kwargs.items()])
            title = f'My Modules {plot_type} for ({plot_kwargs_title})'
            fig = plot_fn(df, **plot_kwargs, title=title)    
            st.plotly_chart(fig)
        # st.write(kwargs)



ValiSocial.run(__name__)
    