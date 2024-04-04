# Play Module README

Play is a Python module that uses the Commune and Streamlit libraries to create interactive dashboards.

## Key Features
- Load and interactively select different modules from an application using a dropdown
- Calls functions from the selected modules that have been served with interactive widgets for function parameters
- View the detailed schema of the selected function 
- Track execution time and display response status 

## Code Explanation

The Play class has two main methods: `dashboard` and `module2streamlit`.

- `dashboard`: Takes a loaded module and creates a dropdown where you can select functions of the module to run.
  
- `module2streamlit`: Converts a function's schema into interactive widgets that user can fill in the values. It populates the function schema with default parameters and extra defaults using the commune library and types retrieved from the function signature. It creates input widgets according to the type of each parameter using streamlit and allows you to run the function by clicking a button, displaying the response status and latency of the function call.


## Typical usage of Play module

```python
from commune import Module 

class CustomModule(Module):
    pass


if __name__ == '__main__':
    Play.run(__name__)
```

This will create a Streamlit app and allows you to serve and interact with all modules discovered by Commune. As long as your module is a subclass of `c.Module`, it can be loaded, served, and called.

## Note:
- Commune library is used for serving modules, connecting to them, and getting a function's schema.
- Streamlit library is used for creating interactive inputs and showing outputs on a Streamlit server.
- The application must have Streamlit's server requirements satisfied to successfully run this module.
- Primary focus of this module is to inspect and call individual functions from the Commune's served modules.
- Be careful as errors handling is minimal and Streamlit server could fail with incorrect inputs.
- The Play class is not meant to be instantiated but used class-wide.
- Ensure Python typing is used in function definitions of the served modules for type-specific widgets to be created in Streamlit.