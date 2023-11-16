import commune as c

class Plotly(c.Module):

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config.sup)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    @classmethod
    def treemap( cls,
                labels:list = ['Category A', 'Category B', 'Category C', 'Other'] ,
                values:list = [10.1, 9.71, 9.36, 71.83], 
                title:str = 'Treemap', 
                font_size:int = 40):
        import plotly.express as px
        import pandas as pd

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

        # Show the
    @classmethod
    def pie( cls,
            labels:list = ['Category A', 'Category B', 'Category C', 'Other'] ,
            values:list = [10.1, 9.71, 9.36, 71.83], 
            title:str = 'Pie Chart', 
            font_size:int = 20, 
            showlegend: int = False):
        import plotly.express as px
        import pandas as pd

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
        return fig.show()

        # Show the
