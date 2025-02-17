

import os
import sys
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
# from commune.plot.dag import DagModule 

import commune as c


class Plot(c.Module):

    height:int=1000
    width:int=1000
    theme: str= 'plotly_dark' 

    @property
    def streamlit_functions(self):
        return [fn for fn in dir(self) if fn.startswith('st_')]  
    
    
    def run(self, data, plots=None, default_plot  ='histogram', title=None ):
        self.cols= st.columns([1,3])
        plots = plots or self.plot_options()
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
                fig = getattr(self, 'plot_'+ plot)(df)
                form.form_submit_button("Render")
        else:
            raise NotImplementedError(f'Broooooo, hold on, you can only use the following {supported_types}')
        fig.update_layout(height=800)
        self.show(fig)
 
    def plot_options(self, prefix:str ='plot_'):
        plot_options = self.fns(prefix)
        return [p.replace(prefix+'_', '')for p in plot_options]


    def show(self, fig):
        with self.cols[1]:
            st.plotly_chart(fig)

    def plot_scatter2D(self, df=None):
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




    def plot_scatter3D(self, df=None):
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


    def plot_box(self, df=None):


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

    def plot_bar(self, df=None):

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




    def plot_histogram(self, df=None):

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


    def plot_heatmap(cls, df=None):

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
        css = r'''
            <style>
                [data-testid="stForm"] {border: 0px}
            </style>
        '''

        st.markdown(css, unsafe_allow_html=True)
        
    @classmethod
    def line_seperator(cls, text='-', length=50):
        st.write(text*length)
      

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
                        v = json.loads(v)
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

    @classmethod
    def plot_dashboard(cls, df, key='dashboard', x='name', y='emission', select_columns=True):
        import plotly.express as px
        import streamlit as st
        cols = list(df.columns)
        if select_columns:
            cols = st.multiselect('Select Columns', cols, cols, key=key+'multi')
        # bar_chart based on x and y

        if len(df) == 0:
            st.error('You are not staked to any modules')
            return 
        col2idx = {c:i for i,c in enumerate(cols)}
        defult_x_col = col2idx.get(x, 0)
        default_y_col = col2idx.get(y, 1)

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
            