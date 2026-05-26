import base64
import io
import itertools
import threading
import time
from pathlib import Path
from typing import Sequence

from openai import OpenAI
from PIL import Image


class VLMClientPool:
    """Thread-safe round-robin pool of vLLM OpenAI-compatible clients across multiple ports."""

    def __init__(
        self,
        ports: Sequence[int] = (8000, 5000),
        host: str = "localhost",
        model_name: str = "Qwen3.5-27B",
        max_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._clients: list[OpenAI] = [
            OpenAI(base_url=f"http://{host}:{port}/v1", api_key="EMPTY")
            for port in ports
        ]
        self._cycle = itertools.cycle(self._clients)
        self._lock = threading.Lock()

    def _get_client(self) -> OpenAI:
        with self._lock:
            return next(self._cycle)

    @staticmethod
    def _encode_image(image_path: Path, scale: float = 1.0) -> str:
        """Encode image file to base64 data URL, optionally scaled down."""
        suffix = image_path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime = mime_map.get(suffix, "image/jpeg")
        if scale >= 1.0:
            data = image_path.read_bytes()
        else:
            img = Image.open(image_path).convert("RGB")
            new_w = max(32, int(img.width * scale))
            new_h = max(32, int(img.height * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            buf = io.BytesIO()
            save_fmt = "JPEG" if suffix in (".jpg", ".jpeg", ".bmp", ".webp") else "PNG"
            img.save(buf, format=save_fmt)
            data = buf.getvalue()
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path,
    ) -> str:
        """
        Send one image+text request.  Returns raw response text.
        Retries up to max_retries times on any exception.
        """
        image_path = Path(image_path)
        last_exc: Exception | None = None
        scale = 1.0
        for attempt in range(self.max_retries):
            client = self._get_client()
            image_data_url = self._encode_image(image_path, scale)
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_data_url}},
                                {"type": "text", "text": user_prompt},
                            ],
                        },
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    presence_penalty=self.presence_penalty,
                    extra_body={
                        "top_k": self.top_k,
                        "min_p": self.min_p,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                content = response.choices[0].message.content or ""
                return content.strip()
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc)
                if "exceeds" in exc_str and "context length" in exc_str.lower():
                    scale *= 0.5
                    print(
                        f"[VLMClientPool] context-length exceeded, halving image "
                        f"(scale={scale:.3f}) and retrying ..."
                    )
                else:
                    print(f"[VLMClientPool] attempt {attempt + 1}/{self.max_retries} failed: {exc}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(
            f"VLM call failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc

    def call_multi_image(
        self,
        system_prompt: str,
        image_paths: list[Path],
        user_text: str,
    ) -> str:
        """
        Send multiple images + a text message in one user turn.
        Images are inserted in order, followed by the text.
        Retries up to max_retries times on any exception.
        """
        content: list[dict] = [
            {"type": "image_url", "image_url": {"url": self._encode_image(p)}}
            for p in image_paths
        ]
        content.append({"type": "text", "text": user_text})

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            client = self._get_client()
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    presence_penalty=self.presence_penalty,
                    extra_body={
                        "top_k": self.top_k,
                        "min_p": self.min_p,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                result = response.choices[0].message.content or ""
                return result.strip()
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(
            f"VLM multi-image call failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc
