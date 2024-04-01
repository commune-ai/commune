# Commune Package Loop Module

This tutorial presents a Python programming guide on how to utilize the `commune` package to manage the Loop class (a subclass of `c.Module`), which presents methods for printing configurations and displaying a dashboard based on user-selected options.

## Getting Started

Begin the process by importing the `commune` package:

```python
import commune as c
```

## Loop Class Definition

A class named `Loop` is defined with the following methods:

- `__init__`: Initializes the `Loop` class but doesn't contain any code in its body.
- `call`: Accepts two integer parameters `x` and `y` with default values of 1 and 2, respectively. It prints the current configuration and returns the sum of `x` and `y`.
- `dashboard`: Uses the `streamlit` package to display a user interface enabling you to select a loop setting. Depending on the option selected, the dashboard for that `loop` is displayed.

**Note**: This class assumes you have a prior understanding of Python classes and the commune and streamlit packages.

```python
class Loop(c.Module):
    def __init__(self):
        pass 

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    
    def dashboard(self):
        import streamlit as st

        loops = c.loops()
        modules = ['module', 'subspace', 'remote']
        option = st.selectbox('Select a Loop', modules, modules.index('module'), key=f'loop.option')
        if option == 'module':
            loop_name = st.selectbox('Select a Loop', loops, loops.index('loop'), key=f'loop.loop_name')
            loop = c.loop(loop_name)
            loop.dashboard()
        elif option == 'subspace':
            loop_name = st.selectbox('Select a Loop', loops, loops.index('loop'), key=f'loop.loop_name')
            loop = c.loop(loop_name)
            loop.dashboard()
        elif option == 'remote':
            loop_name = st.selectbox('Select a Loop', loops, loops.index('loop'), key=f'loop.loop_name')
            loop = c.loop(loop_name)
            loop.dashboard()
        else:
            raise Exception(f'Invalid option {option}')
```

With the `Loop` class, you can create an object and call the class methods with the provided attributes. Always refer to the official `commune` and `streamlit` documentation to understand the details of each function.

## Disclaimer

Please ensure you have installed the `commune` and `streamlit` packages before running the code. This readme guide assumes that you have basic understanding of Python classes and methods. Always refer to the official Python documentation for further reference or clarification.