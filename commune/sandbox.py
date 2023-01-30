from bittensor import utils
import unittest.mock as mock
from unittest.mock import MagicMock, PropertyMock
import os 
import requests
import urllib
import pytest
import miniupnpc

from commune.utils.network import UPNPCException, upnpc_create_port_map
import commune
import langchain
print(commune.upnpc_create_port_map(50050))