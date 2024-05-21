import commune as c
x = c.call('module/dummy_gen', stream=1)
for i in x:
    print(i,'fam')