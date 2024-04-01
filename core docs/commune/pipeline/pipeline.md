# Pipeline Module with Commune Library

The pipeline module with the commune library allows you to build a sequential pipeline, where each step in the pipeline is a module. This module leverages the simplicity provided by the Commune library to easily manage individual code components/modules, provided they adhere to certain constraints for simplicity.

## Steps

The steps involved in configuring and executing the pipeline include the setup, pipeline construction, execution, and testing. 

### Setup

The pipeline is implemented within a class `PipelineModule`, inheriting from `commune.Module`.

```python
from __future__ import annotations
from typing import List, Dict, Union 
import commune

class PipelineModule(commune.Module):
    def __init__(self, modules:List[str]):
        self.build_pipeline(modules)
```

### Pipeline Construction

The pipeline is built during the initialization, where a list of modules is provided.

```python
    def build_pipeline(self, modules:List[Union[str, Dict]], virtual:bool = True, default_call_fn:str = 'forward'):
        self.pipeline_blocks = []
        for module in modules:
            ...
        return self.pipeline_blocks
```

Each module in the pipeline can be a string (module path) or a dictionary that includes the module path, function name, function arguments, and input mapping.

### Pipeline Execution

The pipeline is executed using the `forward()` method.

```python
    def forward(self, **kwargs):
        from copy import deepcopy
        for block in self.pipeline_blocks:
            ...
            
        return kwargs
```

The method loops over all the blocks that are created during pipeline construction. It uses the kwargs inputs to feed to every function call in the blocks.

### Testing

You can test this functionality using the `test_sequential_pipeline()` method.

```python
    @staticmethod
    def test_sequential_pipeline():
        blocks = [
        {
            'module': 'dataset.text.huggingface',
            'fn': 'sample',
            'kwargs': {'tokenize': False},
         }
         ]

        pipeline = Pipeline(pipeline_blocks)
        st.write(pipeline.run())
```

Here, you can define your own pipeline blocks for testing.

## Execution

Once all the steps are implemented in the `PipelineModule`, it is useful to create a driver code or a test script to run the pipeline. You can use the `__main__` method to run the pipeline.

```python
if __name__ == '__main__':
    import streamlit as st

    # st.write(commune.Module.get_module_python_paths())
    # st.write(commune.Module.simple2import('commune.sandbox.paper'))

    Pipeline.test_sequential_pipeline()
```
