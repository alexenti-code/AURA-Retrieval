from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import json
import os
from urllib import request, error


@dataclass(frozen=True)
class CloudProviderConfig:
    provider: str = "router.ai"
    model: str = ""
    api_key_env: str = "MMI_API_KEY"
    base_url: str = "https://routerai.ru/api/v1"
    timeout_seconds: int = 60
    app_name: str = "AURA-Retrieval/MMI"


class CloudProviderClient:
    def __init__(self, config: CloudProviderConfig):
        self.config = config

    def _require_api_key(self) -> str:
        api_key = os.environ.get(self.config.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Environment variable {self.config.api_key_env} is not set. "
                "Provide your router.ai API key before calling the cloud model."
            )
        return api_key

    def _headers(self, api_key: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self.config.app_name,
            "HTTP-Referer": "https://aura.kim",
            "X-Title": self.config.app_name,
        }

    def _payload(self, system_prompt: str, user_prompt: str, lora_adapter: Optional[str] = None) -> Dict[str, Any]:
        if not self.config.model.strip():
            raise RuntimeError("CloudProviderConfig.model is empty. Set the target cloud model name.")

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }

        if lora_adapter:
            payload["metadata"] = {
                "mmi_lora_adapter": str(Path(lora_adapter)),
                "mmi_provider": self.config.provider,
            }

        return payload

    def call_raw(self, system_prompt: str, user_prompt: str, lora_adapter: Optional[str] = None) -> Dict[str, Any]:
        api_key = self._require_api_key()
        payload = self._payload(system_prompt, user_prompt, lora_adapter=lora_adapter)
        body = json.dumps(payload).encode("utf-8")
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        req = request.Request(endpoint, data=body, headers=self._headers(api_key), method="POST")

        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Cloud provider HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Cloud provider connection error: {exc.reason}") from exc

    def call(self, system_prompt: str, user_prompt: str, lora_adapter: Optional[str] = None) -> str:
        data = self.call_raw(system_prompt, user_prompt, lora_adapter=lora_adapter)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Cloud provider returned no choices: {json.dumps(data, ensure_ascii=False)}")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            if text_parts:
                return "\n".join(p for p in text_parts if p)

        raise RuntimeError(f"Cloud provider returned unsupported content format: {json.dumps(data, ensure_ascii=False)}")
