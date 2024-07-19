import commune as c 

class RedVBlue(c.Module):
    def __init__(self, folder_path='history'):
        self.folder_path = self.resolve_path(folder_path)


    