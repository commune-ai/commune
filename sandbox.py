import commune as c
c.print(c.module('serializer')().serialize(c.df([{'miners': 6, 'valis': 2}])))