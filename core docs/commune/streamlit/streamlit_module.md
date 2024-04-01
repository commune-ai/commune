# StreamlitModule README

This Python script is designed to create data visualizations using the web-based framework Streamlit and the data visualization platform Plotly.

# Dependencies
The code depends on several libraries that are expected to be already installed in your environment, including:

- os : used to interact with the operating system
- sys : used to interact with Python runtime environment
- streamlit as st : to create Streamlit applications
- plotly.graph_objects as go : included in the Plotly library, provides a means for efficiently generating plots for large-scale datasets
- plotly.express as px : a simple interface for creating plots.
- pandas as pd: used to create, manipulate and analyze data stored in dataframes.

# Description

Main class in the script is StreamlitModule which inherits from a module class (imported from commune library). It includes methods for creating various types of plots and displaying them via a Streamlit web interface. The plots are rendered through Plotly library.

# Usage

StreamlitModule provides the methods for generating various types of plots (scatter plots, heatmaps, bars, and histograms) by feeding pandas DataFrame to present them in user-friendly way via web interface. Data must be provided as a pandas DataFrame, and the user can choose the type of plot to render.

Styling and formatting methods are also provided, including the option to load custom CSS and apply it to the streamlit page via the 'load_style' method. 

## Important methods

- `run`: This function first checks whether the provided data matches the supported types, then generates a streamlit form where users can choose the type of plot to render. If executed successfully, the data is plotted and displayed.

- `show`: This function is used to display the generated Plotly figure in the streamlit columns

Every `st_plot_` prefixed method like `st_plot_scatter2D`, `st_plot_scatter3D` etc. corresponds to a specific type of plot. These methods take a DataFrame as an argument and return a Plotly figure.

# Note

This code is relatively complex and uses advanced Python concepts such as decorators and static methods. Familiarity with Python, Streamlit, and Plotly is assumed.