import commune
print = commune.print
lines = commune.cmd('df -h').split('\n')[1:]
for l in lines:
    print(l.split(' ')[-1].strip())