
class TorchSerializer:
    def deserialize(self, data: dict) -> 'torch.Tensor':
        from safetensors.torch import load
        if isinstance(data, str):
            data = self.str2bytes(data)
        data = load(data)
        return data['data']

    def serialize(self, data: 'torch.Tensor') -> 'DataBlock':     
        from safetensors.torch import save
        return save({'data':data}).hex()
    
    def str2bytes(self, data: str, mode: str = 'hex') -> bytes:
        if mode in ['utf-8']:
            return bytes(data, mode)
        elif mode in ['hex']:
            return bytes.fromhex(data)
  