import commune as c
import json
import os

class AgentCondense(c.Module):
    description = "This module is used to find files and modules in the current directory"
    anchor="OUTPUT"
  
    def __init__(self, prompt=None, model=None, anchor=None):
        super().__init__(prompt, model, anchor)
        self.prompt = prompt
        self.model = model
        self.anchor = anchor