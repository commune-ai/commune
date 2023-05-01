import commune
modules = commune.modules('module')
commune.print(commune.call(modules, fn='ls'))