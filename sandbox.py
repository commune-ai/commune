import commune as c

c.module('module')
# class Sandbox(c.Module):
#     def store_something(self, x=1):
#         self.put('something', x)

#     def get_something(self):
#         return self.get('something')


# import time
# import numpy as np
# import pandas as pd
# import streamlit as st

# _LOREM_IPSUM = """
# Lorem ipsum dolor sit amet, **consectetur adipiscing** elit, sed do eiusmod tempor
# incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
# nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
# """


# def stream_data():
#     for word in _LOREM_IPSUM.split(" "):
#         yield word + " "
#         time.sleep(0.02)

#     yield pd.DataFrame(
#         np.random.randn(5, 10),
#         columns=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
#     )

#     for word in _LOREM_IPSUM.split(" "):
#         yield word + " "
#         time.sleep(0.02)
# from streamlit.runtime.scriptrunner import add_script_run_ctx

