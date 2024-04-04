# Readme

The following script is composed of a collection of common needed functions that makes restructuring, querying and dictionary operations easier. It includes functionalities such as deleting a key, verifying if a key exists in a dictionary, inserting and getting values for a key in a dictionary. Furthermore, there are additional operations implemented related to dictionaries deep cloning, hash, even data split, recursion and dictionary type conversion in flat-deep vice versa.

This script also includes library imports for operating system manipulation, time handling, random number generation, JSON & YAML file manipulation, and chunks generation. It also provides utility functions for handling Python modules including getting a module, importing an object from a module, and trying a function multiple times if it fails initially.

Moreover, dictionary helper functions were added to get a flattened view of a nested dictionary, perform a deep-merge operation, default-value fallback, overriding values and conversion to a pandas Dataframe. Utilities to read from and save to JSON/YAML files using both synchronous and asynchronous methods were also included, with options to handle specific file-error gracefully. 

Helper methods to convert dictionary and vice versa for Munch object, which is a dictionary that supports attribute-style access, a la JavaScript, were included too. 

Finally, a function to validate key-value argument dictionary against a list or dictionary of defaults values is provided.
Note: The deep2flat, flat2deep, dict_any functions are intended to provide readable and user-friendly way of managing and handling dictionaries.

Requirements: 
- Python 3.6 and above
- numpy, json, yaml, pandas, munch libraries installed

How to use:
Import the file in your project and use the necessary functions as required. Majority of the functions are self-explanatory. However, few functions such as rm_json, chunk_list, get_module, get_object etc., require a bit of a look into the code to better understand their purpose and usage as per your requirement. Generally, all the functions are well equipped to manage operation on dictionaries and module objects.