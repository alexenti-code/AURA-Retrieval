from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class MMIAdapter:
    base_dir: Path

    def load_system_prelude(self) -> str:
        path = self.base_dir / "prompt" / "system_prelude.md"
        return path.read_text(encoding="utf-8").strip()

    def assemble(self, base_system_prompt: str, user_prompt: str) -> str:
        prelude = self.load_system_prelude()
        parts = [p.strip() for p in [base_system_prompt, prelude, user_prompt] if p and p.strip()]
        return "\n\n".join(parts)


if __name__ == "__main__":
    adapter = MMIAdapter(Path(__file__).resolve().parents[1])
    demo = adapter.assemble(
        "You are a language model operating in a bounded runtime.",
        "User: explain what this interface does."
    )
    print(demo)
