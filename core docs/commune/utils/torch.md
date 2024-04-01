# Torch Utility Functions

This module provides utility functions for manipulating and obtaining information about torch data structures such as Tensors and Dictionaries. 

## Functions

### `torch_batchdictlist2dict(batch_dict_list: List, dim:int=0) -> Dict[str, torch.Tensor]`

This function converts a list of dictionaries into a dictionary along a specified dimension. The dictionary's values are torch tensors.

### `tensor_info_dict(input_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]`

Given an input dictionary, this function returns another dictionary where for each Tensor in the input, it gives its shape, dtype, and device.

### `tensor_dict_shape(input_dict: Dict[str, torch.Tensor]) -> Dict[str, Tuple]`

This function returns a dictionary object where each tensor in input is replaced by its shape.

### `check_distributions(kwargs: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]`

This function takes a dictionary of tensors and returns a dictionary that provides the mean and standard deviation for the tensors.

### `confuse_gradients(model)`

This function alters the gradient of a model by introducing random noise.

### `nan_check(input, key_list=[], root_key='')`

This function checks for NaN values in a nested data structure, and returns the keys of the items which contain NaNs.

### `seed_everything(seed: int) -> None`

This function sets the seeds for multiple sources of randomness such as Python's random module, NumPy and PyTorch to ensure reproducible results. Requires integer seed.

### `get_device_memory()`

This function provides information about the GPU's memory. It uses the nvidia_smi library to obtain information about each device.

### `tensor_dict_info(x:Dict[str, 'torch.Tensor']) -> Dict[str, int]`

This function retrieves information about the dictionary's tensors, like shape, dtype, device, and whether it requires gradients.

## Usage 

These utility functions provide a toolkit for handling and obtaining information about PyTorch data structures, like Tensors and Dictionaries. They are particularly useful for doing operations on batches, checking and manipulating gradients, or seeding randomness for reproducibility in machine learning models built with PyTorch. Also, they provide assistance in debugging by checking distributions and NaN values.