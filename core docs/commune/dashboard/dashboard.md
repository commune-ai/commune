# README

This python script defines a web interface for the _Commune_ library, which is a decentralized autonomous organization (DAO) framework for cooperative decision-making. The script uses the _streamlit_ library to create a dynamic web app, and the _pandas_ and _plotly_ libraries for data manipulation and visualization respectively.

## Functionalities
Upon running the script, a web dashboard will be created. The dashboard includes functionalities, such as staking, unstaking, transferring assets between addresses, and registering a new module (service) in the ecosystem.

## Libraries Imported

- _commune_ : The DAO (Decentralized autonomous organization) library for cooperative decision-making.
- _streamlit_ : An open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.
- _pandas_ : Provides high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
- _streamlit.components.v1_ : Components are a way to extend Streamlit's capabilities by reusing Python APIs to integrate JavaScript-based solutions.
- _plotly_ : A graphing library makes interactive, publication-quality graphs. Examples of how much fun it can be found below.

## How It Works
  
The script starts by setting some basic CSS styles for the streamlit-generated form. A `Dashboard` class, which subclasses `commune.Module`, is defined. The class has an initializer (__init__) that gets the currently logged-in user's status and displays it on the sidebar. 

## Methods

Class `Dashboard` contains multiple utility methods for internal use. The key methods used are:

- `select_key`: This method lets users select their identity. 
- `transfer_dashboard`: This method provides a user interface for transferring funds. 
- `module_dashboard`: This method displays a table of namespace.
- `stake_dashboard`: This method provides the user interface for staking.
- `staking_plot`: This method plots the amount staked on different modules.
- `unstake_dashboard`: This method provides the user interface for unstaking.
- `register_dashboard`: This method provides a form for registering a new module with specified parameters.

To start the dashboard, call the `Dashboard.run()` method with __name__ as an argument.

Please install the dependencies using pip before running the script.

Use the following commands to install:

`pip install pandas plotly streamlit commune`

Then, you can run the script by using:

`streamlit run script.py`