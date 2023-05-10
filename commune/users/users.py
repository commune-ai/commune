
import commune as c
from typing import Dict , Any, List
import streamlit as st
import json

class Users(c.Module):
    
    @classmethod
    def test(cls):
        self = cls()
        c.print(self.get_key('bro'))
    
    def add_user(self, name, auth=None):
        key=self.get_key(f"{name}::")
       
        

if __name__ == "__main__":
    Users.test()
    
    