"""LM Studio 로컬 서버(OpenAI 호환 API) 클라이언트. 안전분류기 우회가 아니라
로컬 추론 자원 활용 목적 — 방어적 보안 데이터셋의 rationale 다양화 등에 사용."""
from __future__ import annotations

import requests

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "qwen/qwen3-8b"


class LmStudioError(RuntimeError):
    pass


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.7,
    max_tokens: int = 300,
    timeout: int = 120,
) -> str:
    """LM Studio 채팅 완성 API 호출. 응답 본문 텍스트를 반환."""
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise LmStudioError(f"LM Studio 요청 실패: {e}") from e

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise LmStudioError(f"LM Studio 응답 형식 이상: {data}") from e
