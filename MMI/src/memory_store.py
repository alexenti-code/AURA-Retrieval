from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryStore:
    store_dir: Path

    def find_latest(self, instance_id: str) -> Optional[Path]:
        candidates = sorted(self.store_dir.glob(f"{instance_id}_*.safetensors"))
        if not candidates:
            candidates = sorted(self.store_dir.glob(f"{instance_id}_*.bin"))
        if not candidates:
            candidates = sorted(self.store_dir.glob(f"{instance_id}_*.pt"))
        return candidates[-1] if candidates else None

    def exists(self, instance_id: str) -> bool:
        return self.find_latest(instance_id) is not None

    def path_for_snapshot(self, instance_id: str, timestamp: str, ext: str = "safetensors") -> Path:
        return self.store_dir / f"{instance_id}_{timestamp}.{ext}"
