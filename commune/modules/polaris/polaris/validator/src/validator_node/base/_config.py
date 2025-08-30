from typing import List

from validator.src.validator_node.base.config import PolarisBaseSettings


class ValidatorNodeSettings(PolarisBaseSettings):
    host: str
    port: int
