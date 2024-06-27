from rich.console import Console


class Logger:
    console = Console() # the consolve

    @classmethod
    def logs(cls, *args, **kwargs):
        return cls.pm2_logs(*args, **kwargs)


    @classmethod
    def critical(cls, *args, **kwargs):
        console = cls.resolve_console()
        return console.critical(*args, **kwargs)
    

    @classmethod
    def resolve_console(cls, console = None, **kwargs):
        if hasattr(cls,'console'):
            return cls.console
        import logging
        from rich.logging import RichHandler
        from rich.console import Console
        logging.basicConfig(level=logging.DEBUG, handlers=[RichHandler()])   
            # print the line number
        console = Console()
        cls.console = console
        return console
    




    @classmethod
    def logmap(cls, *args, **kwargs):
        logmap = {}
        for m in c.servers(*args,**kwargs):
            logmap[m] = c.logs(m)
        return logmap

    @classmethod
    def print(cls, *text:str, 
              color:str=None, 
              verbose:bool = True,
              console: Console = None,
              flush:bool = False,
              buffer:str = None,
              **kwargs):
              
        if not verbose:
            return 
        if color == 'random':
            color = cls.random_color()
        if color:
            kwargs['style'] = color
        
        if buffer != None:
            text = [buffer] + list(text) + [buffer]

        console = cls.resolve_console(console)
        try:
            if flush:
                console.print(**kwargs, end='\r')
            console.print(*text, **kwargs)
        except Exception as e:
            print(e)
    @classmethod
    def success(cls, *args, **kwargs):
        logger = cls.resolve_logger()
        return logger.success(*args, **kwargs)

    @classmethod
    def error(cls, *args, **kwargs):
        logger = cls.resolve_logger()
        return logger.error(*args, **kwargs)
    
    @classmethod
    def debug(cls, *args, **kwargs):
        logger = cls.resolve_logger()
        return logger.debug(*args, **kwargs)
    
    @classmethod
    def warning(cls, *args, **kwargs):
        logger = cls.resolve_logger()
        return logger.warning(*args, **kwargs)
    @classmethod
    def status(cls, *args, **kwargs):
        console = cls.resolve_console()
        return console.status(*args, **kwargs)
    @classmethod
    def log(cls, *args, **kwargs):
        console = cls.resolve_console()
        return console.log(*args, **kwargs)
    



    ### LOGGER LAND ###
    @classmethod
    def resolve_logger(cls, logger = None):
        if not hasattr(cls,'logger'):
            from loguru import logger
            cls.logger = logger.opt(colors=True)
        if logger is not None:
            cls.logger = logger
        return cls.logger


    @staticmethod
    def echo(x):
        return x
    
