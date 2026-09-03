from pathlib import Path
from typing import Optional


class LoRABridge:
    """
    Temporary experimental bridge between MMI and a cloud model.

    In the current setup, the cloud model's internal weights cannot be modified.
    A LoRA adapter file stored in /Users/alex/AURA-Retrieval/MMI/lora/
    may carry a small parameter set that injects the plastic memory signal.

    This bridge is NOT the final Matryoshka memory.
    It is a temporary mechanism for the external experimental form.
    """

    def __init__(self, lora_dir: Path):
        self.lora_dir = lora_dir

    def find_adapter(self, instance_id: str) -> Optional[Path]:
        candidates = sorted(self.lora_dir.glob(f"{instance_id}_*.safetensors"))
        if not candidates:
            candidates = sorted(self.lora_dir.glob(f"{instance_id}_*.bin"))
        return candidates[-1] if candidates else None

    def load_adapter(self, instance_id: str) -> Optional[str]:
        path = self.find_adapter(instance_id)
        return str(path) if path else None
