from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class BaseConfig:
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return getattr(self, key, default)
