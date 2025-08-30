class NumpySerializer:
    
    def serialize(self, data: 'np.ndarray') -> 'np.ndarray':     
        return  self.numpy2bytes(data).hex()

    def deserialize(self, data: bytes) -> 'np.ndarray':     
        if isinstance(data, str):
            data = bytes.fromhex(data)
        return self.bytes2numpy(data)

    def bytes2numpy(self, data:bytes) -> 'np.ndarray':
        import msgpack_numpy
        import msgpack
        output = msgpack.unpackb(data, object_hook=msgpack_numpy.decode)
        return output
    
    def numpy2bytes(self, data:'np.ndarray')-> bytes:
        import msgpack_numpy
        import msgpack
        output = msgpack.packb(data, default=msgpack_numpy.encode)
        return output
    
    @classmethod
    def bytes2str(cls, x, **kwargs):
        return x.hex()
    
    @classmethod
    def str2bytes(cls, data: str, mode: str = 'hex') -> bytes:
        if mode in ['utf-8']:
            return bytes(data, mode)
        elif mode in ['hex']:
            return bytes.fromhex(data)
    