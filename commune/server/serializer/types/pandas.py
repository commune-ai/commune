import json
import pandas as pd

class PandasSerializer:

    def serialize(self, data: pd.DataFrame) -> 'DataBlock':
        data = data.to_json()
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return data

    def deserialize(self, data: bytes) -> pd.DataFrame:
        data = pd.DataFrame.from_dict(json.loads(data))
        print(data)
        return data
    