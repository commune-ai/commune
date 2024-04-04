# Plotly Module README

Plotly is a Python module that leverages the Commune and Plotly Express library to generate interactive charts such as Treemaps, Pie Charts, Histograms, and more. 

## Key Features 
- Generate interactive charts using Plotly Express, with support for various chart types including treemaps, pie charts, histograms, and others.
- Display configurations with detailed print statements.
- Allows users to customize chart features like labels, values, and title.

## Code Explanation

The Plotly class encapsulates the creation of four different types of interactive charts. It's a Commune `Module` subclass and has multiple methods for generating various chart types including Treemaps, Pie Charts, Histograms, and other chart types:

- `call`: A simple demonstration method that prints out the configuration objects and returns a summation of two numbers.
- `treemap`: Generates an interactive treemap chart using Plotly Express, with customizable labels, values, and title.
- `plot`: Demonstrates a customizable Plotly Express chart, with chart type as an input parameter.
- `pie`: Generates an interactive Pie Chart using Plotly Express, with customizable labels, values, title and legend visibility.
- `histogram`: Demonstrates another customizable Plotly Express chart.

## Typical usage of Plotly module

The Plotly class can be instantiated and each method can be called with the required parameters.

```python
p = Plotly()
p.call(x=5, y=10)
# Prints configuration and returns 15

p.treemap(labels=["A", "B"], values=[50, 100], title="Sample Treemap")
# Returns treemap chart

p.pie(labels=["A", "B"], values=[50, 100], title="Sample Pie Chart", showlegend=True)
# Returns pie chart

p.plot("scatter")
# Displays a scatter plot with iris data

p.histogram("box")
# Displays a box plot with iris data
```

## Note:
- This module requires [Plotly Express](https://plotly.github.io/plotly.py-docs/generated/plotly.express.html) library for generating different types of charts.
- This module uses the Commune library to serve as a module which can be distributed and communicated.