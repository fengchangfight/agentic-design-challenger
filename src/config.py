import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "llm.yaml"

_config_cache = None


def load_config(path=None):
    global _config_cache
    if path is None and _config_cache is not None:
        return _config_cache
    if path is None:
        path = DEFAULT_CONFIG_PATH
    with open(str(path), "r", encoding="utf-8") as f:
        _config_cache = yaml.safe_load(f)
    return _config_cache
