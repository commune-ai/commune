import commune as c
import plotly.express as px
import pandas as pd


class Plotly(c.Module):

    @classmethod
    def treemap( cls,
                labels:list = ['Category A', 'Category B', 'Category C', 'Other'] ,
                values:list = [10.1, 9.71, 9.36, 71.83], 
                title:str = 'Treemap', 
                font_size:int = 40):

        # Assuming we have the same dataset as before
        data = {
            'labels': labels,
            'values': values
        }

        data = pd.DataFrame(data)

        # Create the treemap
        fig = px.treemap(
            data,
            path=[px.Constant("all"), 'labels'],
            values=values,
            title=title
        )

        # Hide the main "all" label
        fig.data[0].textinfo = 'label+value'
        # increase the font size
        fig.data[0].textfont = {'size': font_size}
        return fig
    

    def plot(self, df=None, chart="scatter", x="sepal_width", y="sepal_length") -> int:
        df = df or px.data.iris()
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        plot = getattr(px, chart)
        fig.show()
        # Show the
    @classmethod
    def pie( cls,
            labels:list = ['Category A', 'Category B', 'Category C', 'Other'] ,
            values:list = [10.1, 9.71, 9.36, 71.83], 
            title:str = 'Pie Chart', 
            font_size:int = 20, 
            showlegend: int = False):
        # Assuming we have the same dataset as before
        data = {
            'labels': labels,
            'values': values
        }

        data = pd.DataFrame(data)

        # Create the treemap
        fig = px.pie(
            data,
            values=values,
            names=labels,
            title=title
        )

        # Hide the main "all" label
        fig.data[0].textinfo = 'label+value'
        # increase the font size
        fig.data[0].textfont = {'size': font_size}
        # remove legend
        fig.update_layout(showlegend=showlegend)
        return fig

        # Show the
    
    def histogram(self, df=None, chart="scatter") -> int:
        df = px.data.iris()
        plot = getattr(px, chart)
        fig = plot(df, x="sepal_width", y="sepal_length")
        return fig


    def app(self):
        import streamlit as st

        c.print('This is the app')
        self.plot()
        self.pie()
        self.treemap()
        self.histogram()
        return 1
    
    def plots(self) -> int:
        fns = dir(px)
        ignore_names = ['Constant', 'IdentityMap']
        fns = [fn for fn in fns if not fn.startswith('_') and not ' ' in fn and not 'utils' in fn]
        return fns




        
