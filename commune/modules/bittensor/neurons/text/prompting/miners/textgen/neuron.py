# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import openai
import argparse
import bittensor
from typing import List, Dict
import os
import commune as c

class TextGenMiner( bittensor.BasePromptingMiner ):

    def __init__( self , config):
        super( TextGenMiner, self ).__init__(config=config)
        self.textgen = c.module('text_generator')()

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass
    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        pass

    def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ) -> str: pass


    def forward( self, messages: List[Dict[str, str]]  ) -> str:

        msg = c.python2str(messages)
        try:
            prompt = f'Given:\n{msg}\n\n Generate a response:\n'

            c.print('Length of prompt: ', len(prompt)) 
            resp = self.textgen.generate(prompt, timeout=8, max_new_tokens=100)
        except Exception as e:
            c.print('Error generating response')
            c.print('\n INPUT ',messages)
            raise e
                
        return resp

if __name__ == "__main__":
    bittensor.utils.version_checking()
    OpenAIMiner().run()