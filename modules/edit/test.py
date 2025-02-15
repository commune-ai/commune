import commune as c
class Test(c.Module):
    def __init__(self):
        super().__init__()
    
    def test(self) -> bool:
        """Test the module"""
        self.print('Testing Edit Module')
        return True
    
    def get_filepath(self) -> str:
        """Return the full file path of this module"""
        return __file__