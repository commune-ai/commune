# Folder Module

This is a Python module used to manage and manipulate files and folders on your system using the Commune package.

## Features

- File Reading: The `file2text` method reads all files from a given folder and returns them as dictionary entries with file names as keys and their respective content as values. 

- File Copying: The `cp` method works like the traditional Unix `cp` command. It copies all files from a specified directory to another one.

## Usage

You can use this module to manage files in any folder of a system that supports Python and has the Commune package. Here is an example:

```python
from Folder import Folder

folder_obj = Folder()
text_dict = folder_obj.file2text('/path/to/folder')
folder_obj.cp('/path/to/folder', '/destination/path')
```

## Key Classes and Methods

- `Folder`: The main class, provides methods for reading from and writing to folders.
- `file2text`: Reads all files from a folder and returns a dictionary that maps file names to their respective content.
- `cp`: Copies all files from one directory to another.
   
## Requirements

- Python
- Commune Python package

## Notice

This code assumes that you have access to the relevant resources and permissions to perform read and write operations at the provided locations. Be aware of potential data overwriting in destination folders when copying files.