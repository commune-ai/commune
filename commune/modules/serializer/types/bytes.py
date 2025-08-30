
class BytesSerializer:

    def serialize(self, data: dict) -> bytes:
        return data.hex()
        
    def deserialize(self, data: bytes) -> 'DataBlock':
        if isinstance(data, str):
            data = bytes.fromhex(data)
        return data
