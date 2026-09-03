from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.runtime import MMIRuntime
from src.memory_store import MemoryStore
from adapters.lora_bridge import LoRABridge


def test_memory_store_returns_none_when_empty():
    store = MemoryStore(BASE_DIR / "memory_store")
    result = store.find_latest("nonexistent_instance")
    assert result is None


def test_lora_bridge_returns_none_when_empty():
    bridge = LoRABridge(BASE_DIR / "lora")
    result = bridge.load_adapter("nonexistent_instance")
    assert result is None


def test_runtime_assembles_with_notice_and_no_local_files():
    runtime = MMIRuntime(BASE_DIR)
    result = runtime.assemble_prompt("Base system", "User prompt", include_notice=True)
    assert "MMI version:" in result
    assert "User prompt" in result
