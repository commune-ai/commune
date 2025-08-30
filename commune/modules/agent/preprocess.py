

import os
import json
import commune as c

class Preprocess:
    def __init__(self, agent='agent'):
        self.agent = c.module(agent)()
   