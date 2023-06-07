
import commune
import numpy as np
from typing import NewType, TypeVar

# image = NewType('image', np.ndarray)
image = TypeVar('image', np.ndarray, np.generic)
label = TypeVar('label', dict, dict[str, any])

class ExampleGradioSchemaBuilder(commune.Module):

    def __init__(self):
        ...

    @staticmethod
    def image_classifier(inp : image) -> label:
        return {'cat': 0.3, 'dog': 0.7}