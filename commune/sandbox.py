import commune as c
import torch
import json
data = c.ticket()
staleness = 1
c.sleep(staleness + 0.1)
print(c.m('ticket')().verify(data, max_age=staleness)) 