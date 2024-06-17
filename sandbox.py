import commune as c
c.print()
px = c.import_module('plotly.express')

class Plotly(c.Module):
    def plots(self, df=None, chart="scatter", x="sepal_width", y="sepal_length") -> int:
        fns = dir(px)
        fns = [fn for fn in fns if not fn.startswith('_') and not ' ' in fn]
        return fns