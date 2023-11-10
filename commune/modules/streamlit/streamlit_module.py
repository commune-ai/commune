

import os
import sys
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
# from commune.plot.dag import DagModule 

import commune as c



class StreamlitModule(c.Module):

    height:int=1000
    width:int=1000
    theme: str= 'plotly_dark' 

    @property
    def streamlit_functions(self):
        return [fn for fn in dir(self) if fn.startswith('st_')]  
    

    def run(self, data, plots=[], default_plot  ='histogram', title=None ):

        self.cols= st.columns([1,3])
        if len(plots) == 0:
            plots = self.plot_options()

        if default_plot not in plots:
            default_plot = plots[0]
        supported_types = [pd.DataFrame]
        if isinstance(data, pd.DataFrame):
            df = data
            with self.cols[1]:
                if len(plots) > 1:
                    name2index = {_name:_idx for _idx, _name in enumerate(plots)}
                    plot = st.selectbox('Choose a Plot', plots, name2index[default_plot])
                else:
                    plot = plots[0]
            form = st.form(F'Params for {plot}')
            with form:
                fig = getattr(self, 'st_plot_'+ plot)(df)
                form.form_submit_button("Render")

        else:
            raise NotImplementedError(f'Broooooo, hold on, you can only use the following {supported_types}')
        fig.update_layout(height=800)
        self.show(fig)
        
    @staticmethod
    def metrics_dict(x, num_rows:int = 1):
        num_elements = len(x)
        num_cols = num_elements//num_rows
        row_cols = [st.columns(num_cols) for i in range(num_rows)]
        for i in range(num_elements):
            k = list(x.keys())[i]
            v = list(x.values())[i]
            row_idx = i//num_cols
            col_idx = i%num_cols
            row_cols[row_idx][col_idx].metric(k, int(v))

    def plot_options(self, prefix:str ='st_plot'):
        plot_options = self.fns(prefix)
        return [p.replace(prefix+'_', '')for p in plot_options]


    def show(self, fig):
        with self.cols[1]:
            st.plotly_chart(fig)

    def st_plot_scatter2D(self, df=None):
        df = df if isinstance(df, pd.DataFrame) else self.df
        column_options = list(df.columns)


        with self.cols[0]:
            st.markdown("## X Axis")
            x_col = st.selectbox("X Axis",column_options, 0 )

            st.markdown("## Y Axis")
            y_col = st.selectbox("Y Axis", column_options, 1)

            st.markdown("## Color Axis")
            color_col = st.selectbox("Color",  column_options + [None],  0)
            color_args = {"color": color_col} if color_col is not None else {}
            marker_size = st.slider("Select Marker Size", 5, 30, 20)

            df["size"] = [marker_size for _ in range(len(df))]

        
        fig = px.scatter(df, x=x_col, y=y_col, size="size", **color_args)
        fig.update_layout(width=1000,
                        height=800)

        return fig




    def st_plot_scatter3D(self, df=None):
        df = df if isinstance(df, pd.DataFrame) else self.df
        column_options = list(df.columns)

        plotly_kwargs = {}
        with self.cols[0]:
            st.markdown("## X Axis")
            plotly_kwargs['x'] = st.selectbox("X Axis", column_options, 0)
            st.markdown("## Y Axis")
            plotly_kwargs['y'] = st.selectbox("Y Axis", column_options, 1)
            st.markdown("## Z Axis")
            plotly_kwargs['z'] = st.selectbox("Z Axis", column_options, 2)
            st.markdown("## Color Axis")
            plotly_kwargs['color'] = st.selectbox("## Color", [None] + column_options, 0)
            marker_size = st.slider("Select Marker Size", 5, 30, 20)
            df["size"] = [marker_size for _ in range(len(df))]
            plotly_kwargs['size']= 'size'
            plotly_kwargs['template'] = self.theme

        fig = px.scatter_3d(df, **plotly_kwargs)
        fig.update_layout(width=self.width, height=self.height, font_size=15)
        return fig


    def st_plot_box(self, df=None):


        df = df if isinstance(df, pd.DataFrame) else self.df
        column_options = list(df.columns)
        plotly_kwargs = {}
        
        with self.cols[0]:
            st.markdown("## X Axis")
            plotly_kwargs['x'] = st.selectbox("X Axis", column_options, 0)
            st.markdown("## Y Axis")
            plotly_kwargs['y'] = st.selectbox("Y Axis", column_options, 1)
            st.markdown("## Color Axis")
            plotly_kwargs['color'] = st.selectbox("Color", [None] + column_options, 0)
            marker_size = st.slider("Select Marker Size", 5, 30, 20)
            df["size"] = [marker_size for _ in range(len(df))]
            plotly_kwargs['template'] = self.theme
            st.markdown("## Box Group Mode")
            plotly_kwargs['boxmode'] = st.selectbox("Choose Box Mode", ["group", "overlay"], 0)

        # df[ plotly_kwargs['x']] = df[ plotly_kwargs['x']].apply(lambda x: str(x)) 
        
        
        fig = px.box(df, **plotly_kwargs)
        fig.update_layout(width=self.width, height=self.height, font_size=20)
        return fig

    def st_plot_bar(self, df=None):

        df = df if isinstance(df, pd.DataFrame) else self.df
        column_options = list(df.columns)


        plot_kwargs = {}
        with self.cols[0]:

            
            st.markdown("## X Axis")
            plot_kwargs['x'] = st.selectbox("X Axis",column_options , 0 )

            st.markdown("## Y Axis")
            plot_kwargs['y'] = st.selectbox("Y Axis", column_options, 0)
            plot_kwargs['barmode'] = st.selectbox("Choose Bar Mode", ["relative", "group", "overlay"], 1)

            st.markdown("## Color Axis")
            plot_kwargs['color'] = st.selectbox("Color",  [None] + column_options, 0 )

        fig = px.bar(df, **plot_kwargs)

        fig.update_layout(width=self.width, height=self.height, font_size=20)
        return fig




    def st_plot_histogram(self, df=None):

        df = df if isinstance(df, pd.DataFrame) else self.df
        column_options = list(df.columns)
        # Choose X, Y and Color Axis
        with self.cols[0]:
            plot_kwargs = {}
            st.markdown("### X-axis")
            plot_kwargs['x'] = st.selectbox("Choose X-Axis Feature", column_options, 0)
            # plot_kwargs['nbins'] = st.slider("Number of Bins", 10, 1000, 10)

            st.markdown("### Y-axis")
            plot_kwargs['y'] = st.selectbox("Choose Y-Axis Feature", [None]+ column_options, 0)

            st.markdown("## Color Axis")
            plot_kwargs['color'] = st.selectbox("Color",  [None]+ column_options , 0 )
            # color_args = {"color":color_col} if color_col is not None else {}
            
            plot_kwargs['barmode'] = st.selectbox("Choose Bar Mode", ["relative", "group", "overlay"], 2)

        

        fig = px.histogram(df, **plot_kwargs)
        fig.update_layout(width=self.width, height=self.height, font_size=20)
        return fig


    def st_plot_heatmap(cls, df=None):

        df = df if isinstance(df, pd.DataFrame) else self.df
        column_options = list(df.columns)
        # Choose X, Y and Color Axis

        plotly_kwargs = {}
        with cls.cols[0]:
            st.markdown("### X-axis")
            plotly_kwargs['x'] = st.selectbox("Choose X-Axis Feature", column_options, 0)
            plotly_kwargs['nbinsx'] = st.slider("Number of Bins", 10, 100, 10)

            st.markdown("### Y-axis")
            plotly_kwargs['y'] = st.selectbox("Choose Y-Axis Feature", [None]+column_options, 0)
            plotly_kwargs['nbinsy'] = st.slider("Number of Bins (Y-Axis)", 10, 100, 10)

            st.markdown("### Z-axis")
            plotly_kwargs['z'] = st.selectbox("Choose Z-Axis Feature", column_options, 0)
            plotly_kwargs['histfunc'] = st.selectbox("Aggregation Function", ["avg", "sum", "min", "sum", "count"], 0)
            plotly_kwargs['template'] = cls.theme

        fig = px.density_heatmap(df, **plotly_kwargs)
        fig.update_layout(width=cls.width, height=cls.height, font_size=20)



        return fig



    
    @classmethod
    def style2path(cls, style:str=None) -> str:
        path = cls.dirpath() + '/styles'
        style2path = {p.split('/')[-1].split('.')[0] : p for p in cls.ls(path)}
        if style != None:
            return style2path[style]
        return style2path
        
        
    @classmethod
    def load_style(cls, style='commune'):
        style_path =  cls.style2path(style)        
        with open(style_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
    @classmethod
    def line_seperator(cls, text='-', length=50):
        st.write(text*length)
      
    @classmethod
    def function2streamlit(cls, 
                           module = None,
                           fn:str = '__init__',
                           fn_schema = None, 
                           extra_defaults:dict=None,
                           cols:list=None,
                           skip_keys = ['self', 'cls'],
                            mode = 'pm2'):
        
        key_prefix = f'{module}.{c.random_word()}'
        if module == None:
            module = cls
            
        elif isinstance(module, str):
            module = c.module(module)
        extra_defaults = {} if extra_defaults is None else extra_defaults
        
        if fn_schema == None:

            fn_schema = module.schema(defaults=True, include_parents=True)[fn]
            if fn == '__init__':
                config = module.config(to_munch=False)
                extra_defaults = config
            kwargs = {}
            fn_schema['default'].pop('self', None)
            fn_schema['default'].pop('cls', None)
            fn_schema['default'].update(extra_defaults)
            fn_schema['default'].pop('config', None)
            fn_schema['default'].pop('kwargs', None)
            
        fn_schema['input'].update({k:str(type(v)).split("'")[1] for k,v in extra_defaults.items()})
        if cols == None:
            cols = [1 for i in list(range(int(len(fn_schema['input'])**0.5)))]
        cols = st.columns(cols)

        for i, (k,v) in enumerate(fn_schema['default'].items()):
            
            optional = fn_schema['default'][k] != 'NA'
            fn_key = k 
            if fn_key in skip_keys:
                continue
            if k in fn_schema['input']:
                k_type = fn_schema['input'][k]
                if 'Munch' in k_type or 'Dict' in k_type:
                    k_type = 'Dict'
                if k_type.startswith('typing'):
                    k_type = k_type.split('.')[-1]
                fn_key = f'**{k} ({k_type}){"" if optional else "(REQUIRED)"}**'
            col_idx  = i 
            if k in ['kwargs', 'args'] and v == 'NA':
                continue
            

            random_word = c.random_word()
            col_idx = col_idx % (len(cols))
            kwargs[k] = cols[col_idx].text_input(fn_key, v, key=f'{key_prefix}.{k}.{random_word}')
     
        kwargs = cls.process_kwargs(kwargs, fn_schema)       
        
        return kwargs

   
    @classmethod
    def process_kwargs(cls, kwargs:dict, fn_schema:dict):
        
        for k,v in kwargs.items():
            if v == 'None':
                v = None
            
            if isinstance(v, str):
                if v.startswith('[') and v.endswith(']'):
                    if len(v) > 2:
                        v = eval(v)
                    else:
                        v = []

                elif v.startswith('{') and v.endswith('}'):

                    if len(v) > 2:
                        v = c.jload(v)
                    else:
                        v = {}               
                elif k in fn_schema['input'] and fn_schema['input'][k] == 'str':
                    if v.startswith("f'") or v.startswith('f"'):
                        v = c.ljson(v)
                    else:
                        v = v

                elif k == 'kwargs':
                    continue
                elif v == 'NA':
                    assert k != 'NA', f'Key {k} not in default'
                elif v in ['True', 'False']:
                    v = eval(v)
                else:
                    v = v
            
            kwargs[k] = v
        return kwargs
    
         
    @classmethod
    def st_metrics_dict(cls, x:str, num_columns=3):
        cols = st.columns(num_columns)
        for i, (k,v) in enumerate(x.items()):
            if type(v) in [int, float, str]:
                cols[i % num_columns].metric(label=k, value=v)
                        
    @classmethod
    def styles(cls):
        return list(cls.style2path().keys())
    
    
    @classmethod
    def style_paths(cls):
        return list(cls.style2path().values())
        

    
    def add_plot_tools(self):
        # sync plots from express
        for fn_name in dir(px):
            if not (fn_name.startswith('__') and fn_name.endswith('__')):
                plt_obj = getattr(px, fn_name)
                if callable(plt_obj):
                    setattr(self, fn_name, plt_obj)

        # self.dag = DagModule()

    @staticmethod
    def local_css(file_name=os.path.dirname(__file__)+'/style.css'):
        import streamlit as st
        
        
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
 
        
    @staticmethod
    def metrics_dict(x, num_rows:int = 1):
        num_elements = len(x)
        num_cols = num_elements//num_rows
        row_cols = [st.columns(num_cols) for i in range(num_rows)]
        for i in range(num_elements):
            k = list(x.keys())[i]
            v = list(x.values())[i]
            row_idx = i//num_cols
            col_idx = i%num_cols
            row_cols[row_idx][col_idx].metric(k, int(v))

    def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:

        from pandas.api.types import (
            is_categorical_dtype,
            is_datetime64_any_dtype,
            is_numeric_dtype,
            is_object_dtype,
        )
        import pandas as pd
        import streamlit as st

        """
        Adds a UI on top of a dataframe to let viewers filter columns

        Args:
            df (pd.DataFrame): Original dataframe

        Returns:
            pd.DataFrame: Filtered dataframe
        """
        modify = st.checkbox("Add filters")

        if not modify:
            return df

        df = df.copy()

        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)

        modification_container = st.container()

        with modification_container:
            to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        min_value=_min,
                        max_value=_max,
                        value=(_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                else:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].astype(str).str.contains(user_text_input)]

        return df