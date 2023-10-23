import commune as c

x = 'fam wahtdup'

c.print(c.submit('module', fn = 'submit',  kwargs=dict(fn='print', args=[x], network='remote',)))