
import torch
import commune as c
print('fammmm')
cmd = 'curl https://api.github.com/repos/commune-ai/commune/contents/README.md \
  -H "Accept: application/json"'
print(c.cmd(cmd))