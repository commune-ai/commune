import commune as c




class Taostats:

    def forward(self,*args, **kwargs):
        return c.ask(*args, process_input=False, **kwargs)