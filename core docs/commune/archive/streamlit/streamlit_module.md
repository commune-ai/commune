# Readme

This script uses Streamlit, Plotly Express and Pandas to construct a Python module called `StreamlitModule` that allows for interactive visualizations of data stored in a Pandas DataFrame. The `StreamlitModule` provides a variety of visualization choices including scatter plots, box plots, bar plots, histograms and heatmaps.

One must create an instance of the `StreamlitModule` class and then use the `run` method of that instance to launch the streamlit user interface, to select a plot type and to customize the plot.

## Prerequisites

- Python 3
- Pandas
- Streamlit
- Plotly Express

## Code Explanation

- The script includes a Python class definition for the StreamlitModule. 
- This class includes methods for creating each type of plot. 
- In each method, users can configure the X, Y, and color axes for the plot.
- With the `run` method you can launch the Streamlit web application and pass in the data you want to visualize.
- The data needs to be in a pandas DataFrame format.
- In the `main` section of the script, an example of using the StreamlitModule class with the iris dataset is provided.

## Quick Start

1. Import the necessary libraries and the `StreamlitModule` class from the script.
2. Create a dataframe for the data you want to visualize.
3. Create an instance of the `StreamlitModule` class.
4. Call the `run` method on the instance, passing in the data and specifying the plot types you want available.

```python
from sklearn.datasets import load_iris
import pandas as pd
from your_script import StreamlitModule    # Replace 'your_script' with the actual name of this script

st_plt = StreamlitModule()
data = load_iris()
df = pd.DataFrame(data=data.data, columns=data.feature_names)

st_plt.run(data=df, plots=['heatmap'])
```

After running the code, a Streamlit application should launch in your default web browser where you can interact with data and visualizations.
