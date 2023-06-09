
from .cli import CLI as cli

if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    cli(args)
