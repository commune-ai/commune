import commune

# commune.launch('commune.metric.MetricServer.MetricServer', name='metric_server',kwargs={'metric': 'ma'})
metrics = commune.connect('metric_server')
print(metrics.module_id)
# print(metrics.set_metric('bob', 5))
# import torch
# metrics.rm_metric('bob')
# adapter_model = commune.get_module('model.adapter')()
# for i in range(10):
#     print(adapter_model.train_model(metric_server='metric_server', tag='base1', refresh=True))
# print(commune.connect('AdapterModel').train_model(num_batches=20, timeout=1))
# print(MetricMap.test())make d