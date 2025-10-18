import requests
import json
from typing import List, Dict, Generator, Optional


class VeniceAI:
    """
    Minimal, dependency-free Venice AI client using direct HTTP requests.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.venice.ai/api/v1",
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ============ Core Helpers ============

    def _post(self, endpoint: str, payload: dict, stream: bool = False):
        url = f"{self.base_url}{endpoint}"
        if stream:
            return requests.post(
                url,
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
        else:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

    # ============ Chat Completion ============

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Send a chat completion request and return the assistant message text.
        Example:
            venice.chat("qwen2.5-32b-instruct", [{"role":"user", "content":"Hello!"}])
        """
        payload = {"model": model, "messages": messages}
        payload.update(kwargs)
        data = self._post("/chat/completions", payload)
        return data["choices"][0]["message"]["content"]

    def chat_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        Stream chat completions token by token.
        Yields text as it arrives.
        """
        payload = {"model": model, "messages": messages, "stream": True}
        payload.update(kwargs)
        with self._post("/chat/completions", payload, stream=True) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line.decode("utf-8"))
                    delta = event["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue

    # ============ Image Generation ============

    def generate_image(
        self,
        model: str,
        prompt: str,
        **kwargs,
    ) -> Dict:
        """
        Generate an image from a text prompt.
        Returns raw JSON (usually includes image URL or base64).
        """
        payload = {"model": model, "prompt": prompt}
        payload.update(kwargs)
        return self._post("/image/generate", payload)

    # ============ Utility ============

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
