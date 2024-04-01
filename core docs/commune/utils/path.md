# Directory Ensuring Function

This function `ensure_dir` checks if a directory for a given file path exists, and if not, it creates one. 

## Function:

### `ensure_dir(file_path: str)`

This function takes a file path as an argument. It uses the `os` library to check if the directory specified in the file path exists and, if it does not, creates it using the `os.makedirs` function.

## Usage:

This function is useful when you have to save files to a directory that might not already exist. Instead of manually checking and creating directories, you can utilize `ensure_dir` to automate these tasks. It aids in smooth and error-free file handling in your code.