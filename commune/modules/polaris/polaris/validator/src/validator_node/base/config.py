from pydantic_settings import BaseSettings
from typing import List


class PolarisBaseSettings(BaseSettings):
    use_testnet: bool = False
    call_timeout: int = 800
