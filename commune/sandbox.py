import inspect

def my_function():
    print("This is my function.")

# Get the source code of the function
source_code = inspect.getsource(my_function)

# Print the source code
print(source_code)