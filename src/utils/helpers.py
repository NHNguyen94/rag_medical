from typing import Dict
from uuid import UUID, uuid4

import yaml


def load_yml_configs(config_path: str) -> Dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_unique_id() -> UUID:
    return uuid4()
