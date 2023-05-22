
# from google.protobuf.json_format import MessageToJson
import commune as c

# c.print(c.module_list())
# # bt = c.module('bittensor')
# # print(bt.get_metagraph())


# # server = c.import_object('commune.bittensor.neuron.text.server')
# # server.serve()
# # print(server)

# c.new_event_loop()
# sample = c.module('dataset').sample()
model = c.connect('server::fish::1')
import torch
print(model.encode_forward_causallmnext(torch.tensor([[0,2,4,5,6]])))
# # print(sample)

# print(c.call_pool('server', 'encode_forward_causallmnext', sample['input_ids'], timeout=3))

# 