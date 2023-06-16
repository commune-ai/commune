import commune as c

for p in range(570, 600):
    c.print(p)
    print(c.cmd(f'pm2 delete {p}'))