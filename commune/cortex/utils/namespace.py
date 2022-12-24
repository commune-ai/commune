class SimpleNamespace:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

class RecursiveNamespace:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)
    for k, v in kwargs.items():
        if isinstance(v, dict):
            self.__dict__[k] = RecursiveNamespace(**v)