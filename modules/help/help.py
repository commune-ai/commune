import commune as c
class Cmdr:
    def forward(self, *cmd):
        return c.ask(*cmd)
