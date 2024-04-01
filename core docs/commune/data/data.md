# README

This script contains a `Dataset` class that subclasses the `Module` class from the `commune` library and `Dataset` class from Pytorch's `torch.utils.data`. Its main purpose is to interact with a chosen dataset, enabling interactions such as sampling from it.

## Key Features

- Configuration of dataset and model for either training or other use-cases
- Sampling of data with cloud functionality enabled by `commune`
- Predefined mode shortcuts for various dataset types
- Sample validity checks
- Control over sampling timeout and retries

## Libraries Used
- `commune`: A Python library for decentralized autonomous systems
- `torch`: A deep learning library
- `asyncio`: A library for writing single-threaded concurrent code 

## Classes and Methods

- `Dataset`: A class which ties the functionalities of `torch.utils.data.Dataset` and `commune.Module`. 

    - `__init__`: Initializes instance of `Dataset`. Accepts parameters for dataset, configuration and additional arguments.

    - `set_dataset`, `set_model` : Configures the dataset and model based on configuration input.

    - `sample_check`: Verifies that a sample meets set criteria.
  
    - `async_sample`: A coroutine that attempts to get a sample from the dataset. It retries if the sample does not pass `sample_check`. If all attempts fail, an exception is raised.

    - `sample`: An interface for calling `async_sample` as a coroutine with timeout support.

Use the class by creating an instance with your specific parameters.

## Usage
  
Ensure the required modules are installed before running the code: `commune`, `torch`, `asyncio`.

This class does not run standalone and should be imported into another script to use. How you choose to interact with it depends on your needs. You can create an instance with your specific parameters then call the `sample` method to receive data from your chosen dataset.

For example, given `dataset='my_dataset', 'config'=my_config`, create a class `instance = Dataset(dataset, config)`. Then, fetch data via `data = instance.sample()`. You may include timeout and retry limits like so `data = instance.sample(timeout=2, retries=3)`.

Tests should be written for the implementation of the class in your script. 

For further information about the libraries used, see:
- [Commune](https://github.com/commune/commune): A Python library for decentralized autonomous systems.
- [Torch](https://pytorch.org/): An open source deep learning platform.
- [AsyncIO](https://docs.python.org/3/library/asyncio.html): A library to write single-threaded concurrent code using coroutines, multiplexing I/O access over sockets and other resources.
