import commune as c

fns = c.parent_functions() + c.child_functions()
for f in c.functions(include_parents=1):
    if f not in fns:
        print(f)