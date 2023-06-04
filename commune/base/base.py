import commune

class Base(commune.Module):
    def __init__(self, **kwargs):
        config = self.set_config(config=kwargs)
    def run(self):
        print('Base run')
    
if __name__ == '__main__':
    Base.run()