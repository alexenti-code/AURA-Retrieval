from pathlib import Path
from typing import Optional
from .adapter import MMIAdapter
from .config import MMIConfig


class MMIRuntime:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.adapter = MMIAdapter(base_dir)
        self.config = MMIConfig.from_json(base_dir / "schema" / "example.anatomy.json")

    def render_memory_notice(self) -> str:
        lines = [
            f"MMI version: {self.config.version}",
            f"Core: {self.config.core_symbol} — {self.config.core_description}",
            f"Interface: {self.config.interface_name} — {self.config.interface_role}",
            f"Memory: {self.config.memory_symbol} — {self.config.memory_description}",
        ]
        if self.config.timescales:
            lines.append("Timescales: " + ", ".join(self.config.timescales))
        return "\n".join(lines)

    def render_local_state_notice(self, memory_path: Optional[str] = None, lora_path: Optional[str] = None) -> str:
        lines = [
            "MMI Local State Notice",
            f"Local memory file: {'present' if memory_path else 'absent'}",
            f"Local memory path: {memory_path if memory_path else 'none'}",
            f"LoRA adapter: {'present' if lora_path else 'absent'}",
            f"LoRA path: {lora_path if lora_path else 'none'}",
            "These are runtime facts, not assumptions.",
            "If a local file is absent, do not claim that it exists.",
        ]
        return "\n".join(lines)

    def assemble_prompt(
        self,
        base_system_prompt: str,
        user_prompt: str,
        include_notice: bool = False,
        memory_path: Optional[str] = None,
        lora_path: Optional[str] = None,
        include_local_state: bool = False,
    ) -> str:
        system_prompt = base_system_prompt.strip()
        blocks = [system_prompt] if system_prompt else []

        if include_notice:
            blocks.append(self.render_memory_notice())

        if include_local_state:
            blocks.append(self.render_local_state_notice(memory_path=memory_path, lora_path=lora_path))

        final_system_prompt = "\n\n".join(block for block in blocks if block and block.strip())
        return self.adapter.assemble(final_system_prompt, user_prompt)


if __name__ == "__main__":
    runtime = MMIRuntime(Path(__file__).resolve().parents[1])
    print(runtime.assemble_prompt(
        "You are a language model operating in a bounded runtime.",
        "User: describe the current role of MMI.",
        include_notice=True,
        include_local_state=True,
    ))
