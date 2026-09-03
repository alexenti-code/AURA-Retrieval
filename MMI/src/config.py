from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import json


@dataclass(frozen=True)
class MMIConfig:
    version: str
    core_symbol: str
    core_description: str
    interface_name: str
    interface_role: str
    non_functions: List[str] = field(default_factory=list)
    memory_symbol: str = "Φ(t)"
    memory_description: str = "separate plastic autobiographical parameter body"
    timescales: List[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: Path) -> "MMIConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            version=data["version"],
            core_symbol=data["core"]["symbol"],
            core_description=data["core"]["description"],
            interface_name=data["memory_interface"]["name"],
            interface_role=data["memory_interface"]["role"],
            non_functions=list(data["memory_interface"].get("non_functions", [])),
            memory_symbol=data["memory_body"]["symbol"],
            memory_description=data["memory_body"]["description"],
            timescales=list(data["memory_body"].get("timescales", [])),
        )
