import base64
import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from lib.sun397_filtering import ImageCandidate


SYSTEM_PROMPT = (
    "You are a strict visual filter for research dataset curation. "
    "Decide whether an image is a suitable background for natural human face compositing. "
    "Return JSON only."
)

USER_PROMPT = """Evaluate this image for natural human face compositing background suitability.

Decision target:
- keep: the background is realistic and usable for natural face compositing.
- reject: the background is unsuitable.

Reject if any of the following dominates the image:
- visible real person or clear human face
- mirror selfie or strong identity-bearing reflection
- privacy-sensitive or information-dense text/signals
- foreground subject dominance
- geometry, perspective, lighting, clutter, or occlusion that would likely make face compositing look unnatural

Return JSON only with this schema:
{
  "decision": "keep" | "reject",
  "reason": "short reason in English"
}
"""


@dataclass
class VLMResult:
    # VLM       
    image_path: str
    split: str
    class_name: str
    width: int
    height: int
    pixel_count: int
    decision: str
    reason: str
    model: str
    raw_response: str
    error: str | None = None


class QwenVLMClient:
    #  vLLM OpenAI  API    .
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        timeout_seconds: float = 180.0,
        temperature: float = 0.7,
        top_p: float = 0.8,
        presence_penalty: float = 1.5,
        top_k: int = 20,
        min_p: float = 0.0,
        max_tokens: int = 256,
    ):
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)

    def _encode_image(self, image_path: Path) -> str:
        #   data URL  VLM  .
        mime_type = "image/jpeg"
        suffix = image_path.suffix.lower()
        if suffix == ".png":
            mime_type = "image/png"
        elif suffix == ".webp":
            mime_type = "image/webp"
        elif suffix == ".bmp":
            mime_type = "image/bmp"

        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _parse_response_text(self, response_text: str) -> tuple[str, str]:
        #  JSON     decision reason .
        normalized_text = response_text.strip()
        if normalized_text.startswith("```"):
            normalized_text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", normalized_text)
            normalized_text = re.sub(r"\s*```$", "", normalized_text)

        try:
            parsed = json.loads(normalized_text)
            decision = str(parsed.get("decision", "reject")).strip().lower()
            reason = str(parsed.get("reason", "no_reason")).strip()
            if decision not in {"keep", "reject"}:
                decision = "reject"
            return decision, reason or "no_reason"
        except json.JSONDecodeError:
            start = normalized_text.find("{")
            end = normalized_text.rfind("}")
            if start != -1 and end != -1 and start < end:
                try:
                    parsed = json.loads(normalized_text[start : end + 1])
                    decision = str(parsed.get("decision", "reject")).strip().lower()
                    reason = str(parsed.get("reason", "no_reason")).strip()
                    if decision not in {"keep", "reject"}:
                        decision = "reject"
                    return decision, reason or "no_reason"
                except json.JSONDecodeError:
                    pass

        decision_match = re.search(r'"decision"\s*:\s*"(keep|reject)"', normalized_text, flags=re.IGNORECASE)
        reason_match = re.search(r'"reason"\s*:\s*"([^\"]*)', normalized_text, flags=re.IGNORECASE)

        if decision_match:
            decision = decision_match.group(1).lower()
        else:
            lowered = normalized_text.lower()
            decision = "keep" if '"decision": "keep"' in lowered or "keep" in lowered[:80] else "reject"

        if reason_match:
            reason = reason_match.group(1).strip() or "no_reason"
        else:
            reason = normalized_text or "unparseable_response"

        return decision, reason

    def _extract_message_text(self, response) -> str:
        # content         .
        message = response.choices[0].message
        content = getattr(message, "content", None)

        if isinstance(content, str) and content.strip():
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                elif hasattr(item, "text"):
                    text_parts.append(str(item.text))
            joined = "\n".join(part for part in text_parts if part)
            if joined.strip():
                return joined

        reasoning = getattr(message, "reasoning", None)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning

        return ""

    def classify_image(self, candidate: ImageCandidate) -> VLMResult:
        #    Qwen  .
        image_path = Path(candidate.image_path)
        image_data_url = self._encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url,
                            },
                        },
                        {
                            "type": "text",
                            "text": USER_PROMPT,
                        },
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
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
        )

        raw_response = self._extract_message_text(response)
        decision, reason = self._parse_response_text(raw_response)

        return VLMResult(
            image_path=candidate.image_path,
            split=candidate.split,
            class_name=candidate.class_name,
            width=candidate.width,
            height=candidate.height,
            pixel_count=candidate.pixel_count,
            decision=decision,
            reason=reason,
            model=self.model_name,
            raw_response=raw_response,
        )


def classify_candidates_stream(
    client: QwenVLMClient,
    candidates: list[ImageCandidate],
    concurrency: int,
    retry_count: int,
    progress_callback: Callable[[dict[str, int], VLMResult | None], None] | None = None,
) -> list[VLMResult]:
    #    ,         .
    if concurrency <= 0:
        raise ValueError("concurrency 1  .")

    results: list[VLMResult] = []
    pending: dict[Future, tuple[ImageCandidate, int]] = {}
    iterator = iter(candidates)
    total_count = len(candidates)
    submitted_count = 0
    completed_count = 0
    keep_count = 0
    reject_count = 0
    failed_count = 0
    last_heartbeat_at = 0.0

    def emit_progress(result: VLMResult | None = None) -> None:
        #        tqdm     .
        if progress_callback is None:
            return

        progress_callback(
            {
                "total": total_count,
                "submitted": submitted_count,
                "completed": completed_count,
                "keep": keep_count,
                "reject": reject_count,
                "failed": failed_count,
                "in_flight": len(pending),
            },
            result,
        )

    def submit_candidate(executor: ThreadPoolExecutor, candidate: ImageCandidate, attempt: int) -> None:
        nonlocal submitted_count
        future = executor.submit(client.classify_image, candidate)
        pending[future] = (candidate, attempt)
        if attempt == 0:
            submitted_count += 1
        emit_progress()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for _ in range(concurrency):
            try:
                candidate = next(iterator)
            except StopIteration:
                break
            submit_candidate(executor, candidate, 0)

        emit_progress()

        while pending:
            done, _ = wait(pending.keys(), timeout=1.0, return_when=FIRST_COMPLETED)

            if not done:
                now = time.monotonic()
                if now - last_heartbeat_at >= 10.0:
                    emit_progress()
                    last_heartbeat_at = now
                continue

            for future in done:
                candidate, attempt = pending.pop(future)

                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    if result.decision == "keep":
                        keep_count += 1
                    else:
                        reject_count += 1
                    emit_progress(result)
                except Exception as exc:
                    if attempt < retry_count:
                        submit_candidate(executor, candidate, attempt + 1)
                        continue

                    failed_result = VLMResult(
                        image_path=candidate.image_path,
                        split=candidate.split,
                        class_name=candidate.class_name,
                        width=candidate.width,
                        height=candidate.height,
                        pixel_count=candidate.pixel_count,
                        decision="reject",
                        reason="request_failed",
                        model=client.model_name,
                        raw_response="",
                        error=str(exc),
                    )
                    results.append(failed_result)
                    completed_count += 1
                    reject_count += 1
                    failed_count += 1
                    emit_progress(failed_result)

                try:
                    next_candidate = next(iterator)
                    submit_candidate(executor, next_candidate, 0)
                except StopIteration:
                    pass

        emit_progress()

    return results


def vlm_result_to_dict(result: VLMResult) -> dict[str, Any]:
    # dataclass  JSON   dict .
    return asdict(result)
