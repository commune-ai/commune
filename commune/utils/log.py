
import time
import sys
import itertools
from contextlib import contextmanager

# As a context manager
@contextmanager
def print_load(message="Loading", duration=5):
    spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    factor = 4
    
    # Create a flag to control the animation
    stop_animation = False
    
    def animate():
        start_time = time.time()
        try:
            while not stop_animation:
                for frame in spinner:
                    if stop_animation:
                        break
                    loading_text = f"\r{CYAN}{frame*factor}{message}({int(time.time() - start_time)}s){frame*factor}"
                    sys.stdout.write(loading_text)
                    sys.stdout.flush()
                    time.sleep(0.1)
        except KeyboardInterrupt:
            sys.stdout.write("\r" + " " * (len(message) + 10))
            sys.stdout.write(f"\r{CYAN}Loading cancelled!{RESET}\n")
    
    # Start animation in a separate thread
    import threading
    thread = threading.Thread(target=animate)
    thread.start()
    
    try:
        yield
    finally:
        # Stop the animation
        stop_animation = True
        thread.join()
        sys.stdout.write("\r" + " " * (len(message) + 10))
        sys.stdout.write(f"\r{CYAN}✨ {message} complete!{RESET}\n")

def test_loading_animation():
    with print_load("Testing", duration=3):
        time.sleep(3)

    


def resolve_console( console = None, **kwargs):
    import logging
    from rich.logging import RichHandler
    from rich.console import Console
    logging.basicConfig( handlers=[RichHandler()])   
        # print the line number
    console = Console()
    console = console
    return console

def print_console( *text:str, 
            color:str=None, 
            verbose:bool = True,
            console: 'Console' = None,
            flush:bool = False,
            buffer:str = None,
            **kwargs):
            
    if not verbose:
        return 
    if color == 'random':
        color = random_color()
    if color:
        kwargs['style'] = color
    
    if buffer != None:
        text = [buffer] + list(text) + [buffer]

    console = resolve_console(console)
    try:
        if flush:
            console.print(**kwargs, end='\r')
        console.print(*text, **kwargs)
    except Exception as e:
        print(e)

def success( *args, **kwargs):
    logger = resolve_logger()
    return logger.success(*args, **kwargs)

def error( *args, **kwargs):
    logger = resolve_logger()
    return logger.error(*args, **kwargs)


def debug( *args, **kwargs):
    logger = resolve_logger()
    return logger.debug(*args, **kwargs)

def warning( *args, **kwargs):
    logger = resolve_logger()
    return logger.warning(*args, **kwargs)

def status( *args, **kwargs):
    console = resolve_console()
    return console.status(*args, **kwargs)

def log( *args, **kwargs):
    console = resolve_console()
    return console.log(*args, **kwargs)

### LOGGER LAND ###

def resolve_logger( logger = None):
    if not hasattr('logger'):
        from loguru import logger
        logger = logger.opt(colors=True)
    if logger is not None:
        logger = logger
    return logger


def critical( *args, **kwargs):
    console = resolve_console()
    return console.critical(*args, **kwargs)

def echo(x):
    return x