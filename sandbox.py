import commune
modules = commune.modules('model.gptj::huck')
model = commune.module('model.transformer')
model.generate('wHAT IS THE POINT OF LIFE?', model='model.gptj::huck')
