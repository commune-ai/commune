# README

This script provides a `Tree` class that interacts with the Commune module, allowing the user to manage and browse a file tree of their project modules. The `Tree` class is built as a Commune subclass and maintains a map of the entire project structure. 

## Usage

1. Import necessary libraries and the script.
2. To use, create an instance of the `Tree` class. This class provides the explicit structure of your project and provides methods for scanning and updating your files.
3. `Tree` includes several class-level functions such as `add_tree`, `build_tree`, `get_module_python_paths`, `get_tree_root_dir`, and `path2simple`.
4. An instance of `Tree` is able to scan your project and create a representation of your file hierarchy in a clear and concise way.

## Functionality

* add_tree: Registers the path of a new subtree in the tree.
* build_tree: Builds the project's module tree and optionally updates it. 
* get_module_python_paths: Searches for all modules with YAML files and returns their paths. 
* get_tree_root_dir: Retrieves the root directory of the tree structure from its state.
* path2simple: Converts a given path into a simple path string for an easy readability.

## Dependencies

* Commune - A Python library for managing distributed file systems.
* os - A Python module that provides a way of using operating system dependent functionality.
* glob - A Python module to find all the pathnames matching a specified pattern.

These can be installed using pip.

## Run

The script can be run using a Python environment by calling `python scriptname.py`.

## Note

Make sure files necessary for building the tree structure are in the correct format and path. These files are required for functions such as `get_module_python_paths` and `add_tree`. The script assumes that the Commune module has been correctly installed and set up in your environment.