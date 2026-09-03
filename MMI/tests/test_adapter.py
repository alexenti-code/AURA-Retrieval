from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.adapter import MMIAdapter
from src.runtime import MMIRuntime


def test_adapter_assembles_parts():
    adapter = MMIAdapter(BASE_DIR)
    result = adapter.assemble("Base system", "User prompt")
    assert "Base system" in result
    assert "User prompt" in result
    assert "autobiographical memory exists" in result


def test_runtime_loads_schema_and_notice():
    runtime = MMIRuntime(BASE_DIR)
    notice = runtime.render_memory_notice()
    assert "MMI version:" in notice
    assert "Core: K" in notice
    assert "Memory: Φ(t)" in notice


def test_runtime_can_embed_notice():
    runtime = MMIRuntime(BASE_DIR)
    result = runtime.assemble_prompt("Base system", "User prompt", include_notice=True)
    assert "MMI version:" in result
    assert "User prompt" in result
