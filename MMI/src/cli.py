from pathlib import Path
import argparse
import json
import os
import sys
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.runtime import MMIRuntime
from src.memory_store import MemoryStore
from adapters.lora_bridge import LoRABridge
from adapters.cloud_provider import CloudProviderClient, CloudProviderConfig


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def mask_secret(value: str) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def write_run_artifact(runs_dir: Path, instance: str, payload: dict) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = runs_dir / f"{instance}_{ts}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main():
    load_env_file(BASE_DIR / ".env")

    parser = argparse.ArgumentParser(description="MMI CLI — assemble a memory-aware prompt and optionally call a cloud model")
    parser.add_argument("--instance", default="aura_instance_001", help="Instance ID for memory lookup")
    parser.add_argument("--base-prompt", default="", help="Base system prompt for the model")
    parser.add_argument("--user-prompt", required=True, help="User prompt or question")
    parser.add_argument("--notice", action="store_true", help="Include MMI memory notice in the assembled prompt")
    parser.add_argument("--local-state", action="store_true", default=True, help="Include strict local state notice in the assembled prompt")
    parser.add_argument("--lora", action="store_true", help="Check for a local LoRA adapter for this instance")
    parser.add_argument("--dry-run", action="store_true", help="Print the assembled prompt without calling the provider")
    parser.add_argument("--call-provider", action="store_true", help="Call the configured cloud provider")
    parser.add_argument("--save-run", action="store_true", help="Save the request/response metadata to runs/")
    parser.add_argument("--model", default=os.environ.get("MMI_MODEL", ""), help="Provider model name")
    parser.add_argument("--base-url", default=os.environ.get("MMI_BASE_URL", "https://routerai.ru/api/v1"), help="Provider API base URL")
    args = parser.parse_args()

    runtime = MMIRuntime(BASE_DIR)
    store = MemoryStore(BASE_DIR / "memory_store")
    lora = LoRABridge(BASE_DIR / "lora")

    memory_path = store.find_latest(args.instance)
    lora_path = lora.load_adapter(args.instance) if args.lora else None
    assembled = runtime.assemble_prompt(
        args.base_prompt,
        args.user_prompt,
        include_notice=args.notice,
        memory_path=str(memory_path) if memory_path else None,
        lora_path=str(lora_path) if lora_path else None,
        include_local_state=args.local_state,
    )

    print("=== MMI Assembly ===")
    print(assembled)
    print()
    print("=== Local State ===")
    print(f"Memory file : {memory_path or 'none found'}")
    print(f"LoRA adapter: {lora_path or 'none found'}")
    print(f"Provider URL: {args.base_url}")
    print(f"Provider model: {args.model or 'none set'}")
    print(f"API key: {mask_secret(os.environ.get('MMI_API_KEY', ''))}")

    if args.call_provider:
        config = CloudProviderConfig(model=args.model, base_url=args.base_url)
        client = CloudProviderClient(config)
        print()
        print("=== Provider Response ===")
        response = client.call(system_prompt=assembled, user_prompt=args.user_prompt, lora_adapter=str(lora_path) if lora_path else None)
        print(response)

        if args.save_run:
            artifact = write_run_artifact(
                BASE_DIR / "runs",
                args.instance,
                {
                    "instance": args.instance,
                    "base_url": args.base_url,
                    "model": args.model,
                    "memory_file": str(memory_path) if memory_path else None,
                    "lora_adapter": str(lora_path) if lora_path else None,
                    "user_prompt": args.user_prompt,
                    "response": response,
                },
            )
            print()
            print(f"Saved run: {artifact}")


if __name__ == "__main__":
    main()
