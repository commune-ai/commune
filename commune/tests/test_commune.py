
import pytest
import commune as c 


def test_key():
    c.module('key').test()

def test_namespace():
    c.module('namespace').test()

def test_server():
    c.module('server').test()

def test_subnet():
    c.module('subnet').test()
