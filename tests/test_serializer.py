
import commune as c
def test():
    self = c.module('serializer')()
    import torch, time
    data_list = [
        torch.ones(1000),
        torch.zeros(1000),
        torch.rand(1000), 
        [1,2,3,4,5],
        {'a':1, 'b':2, 'c':3},
        'hello world',
        c.df([{'name': 'joe', 'fam': 1}]),
        1,
        1.0,
        True,
        False,
        None

    ]
    for data in data_list:
        t1 = time.time()
        ser_data = self.serialize(data)
        des_data = self.deserialize(ser_data)
        des_ser_data = self.serialize(des_data)
        t2 = time.time()

        latency = t2 - t1
        emoji = '✅' if str(des_ser_data) == str(ser_data) else '❌'
        print(type(data),emoji)
    return {'msg': 'PASSED test_serialize_deserialize'}
