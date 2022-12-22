def merge_objects(self, self2, functions:list):
    self.fn_signature_map = {}
    for fn_key in functions:
        def fn( *args, **kwargs):
            self2_fn = getattr(self2, fn_key)

            return self2_fn(*args, **kwargs)
        setattr(self, fn_key, partial(fn, self))