
import commune as c
import os
from typing import Dict, Any, Optional, Union

class RmFile:
    
    def forward(self, 
                path: str): 
        """Remove a file at the specified path.
        Args:
            path (str): The path to the file to be removed.
        """
        abs_path = os.path.abspath(path)
        if os.path.isfile(abs_path):
            os.remove(abs_path)
            return f"File '{abs_path}' has been removed."
        else:
            return f"File '{abs_path}' does not exist."

    def test(self, file_path = './test.txt'):
        c.put_text('./test.txt', 'This is a test file.')
        assert os.path.isfile(file_path)
        result = self.forward('./test.txt')
        assert not os.path.isfile(os.path.abspath('./test.txt'))
        return {"success": True, "message": result}


                