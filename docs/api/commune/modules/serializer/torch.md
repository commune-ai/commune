# torch



Source: `commune/modules/serializer/torch.py`

## Classes

### TorchSerializer



#### Methods

##### `deserialize(self, data: dict) -> 'torch.Tensor'`



Type annotations:
```python
data: <class 'dict'>
return: torch.Tensor
```

##### `serialize(self, data: 'torch.Tensor') -> 'DataBlock'`



Type annotations:
```python
data: torch.Tensor
return: DataBlock
```

##### `str2bytes(self, data: str, mode: str = 'hex') -> bytes`



Type annotations:
```python
data: <class 'str'>
mode: <class 'str'>
return: <class 'bytes'>
```

