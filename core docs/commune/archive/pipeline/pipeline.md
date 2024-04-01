# PipelineModule for Commune

This Python script introduces the `PipelineModule` for Commune, a powerful framework that simplifies the process of building and executing complex pipelines.

## Overview

The `PipelineModule` class is responsible for building a pipeline using a list of modules to transform the input data step by step. It manages transformations through blocks, where each block is a dict containing a function (`fn`) from a module along with its associated input map (`input_map`) and keyword arguments (`kwargs`).

The `forward` method applies these transformations to the input data in a sequential order and produces the final output after all blocks have been executed.

A test method `test_sequential_pipeline` is provided for quick functionality checks.

## Requirements

- **Python 3**: The script is implemented purely in Python, however, it uses type hint annotations from Python 3.5 or later.
- **Commune**: Commune needs to be installed as this module is based on it.
- **Streamlit**: Streamlit is used in the test module and thus has to be installed.

## Usage

### Instantiate PipelineModule

Create a pipeline module by passing a list of module names:

```python
pipeline = PipelineModule(modules_list)
```
### Build a Pipeline

Build your own pipeline by providing a list of modules:

```python
pipeline.build_pipeline(modules)
```
### Forward Method

Execute your pipeline with the forward method:

```python
pipeline.forward(**kwargs)
```
## Testing

The `test_sequential_pipeline` can be used to test this pipeline. 

**Note:** Please update the `blocks` list in `test_sequential_pipeline` function to fit your testing needs.

## Support

If you encounter issues or require support, please refer to the provided email address or platform for contact.

## License

PipelineModule for Commune is licensed under the MIT license.
