import commune as c

class Demo(c.Module):
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())

    def new_key(self):
        cmd = "openssl req -x509 -newkey rsa:4096 -nodes -out cert .pem -keyout key.pem -days 365"
        return c.cmd(cmd)