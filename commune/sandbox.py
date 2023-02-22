import commune


commune.get_module('model.transformer')


MetricMap = commune.get_module('commune.metric.MetricMap.MetricMap')
print(MetricMap.test())
# print(MetricMap.test())