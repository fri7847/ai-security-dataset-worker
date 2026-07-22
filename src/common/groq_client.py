"""Groq 무료 티어(OpenAI 호환 API) 클라이언트.

GitHub Actions 등 사용자 로컬 PC 자원을 쓰지 않는 실행 환경에서 rationale 생성 등에
사용하기 위한 백엔드. API 키는 환경변수 GROQ_API_KEY로 전달한다(코드에 하드코딩 금지).
무료 티어 한도(실측, 2026-07-22): RPM 30, **TPM 12,000**(검색으로 찾은 6K는 부정확했음 —
실제 429 에러 메시지의 Limit 값으로 재확인). 소규모 배치(20건, workers=2)로 실측한 결과
TPM 한도가 RPM보다 먼저 걸려 45%가 429로 실패함 — diversify_rationale.py는 실패를
"kept_original"로 기록해 --resume 시 영구 스킵하므로, 여기서 재시도 없이 그냥 실패시키면
안 됨. 429 응답의 "Please try again in Xs" 문구를 파싱해 그만큼 대기 후 재시도한다.
"""
from __future__ import annotations

import os
import re
import time

import requests

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 6

RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s")


def _parse_retry_after(resp) -> float:
    header = resp.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    m = RETRY_AFTER_RE.search(resp.text)
    if m:
        return float(m.group(1))
    return 8.0


class GroqError(RuntimeError):
    pass


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.4,
    max_tokens: int = 300,
    timeout: int = 60,
) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise GroqError("GROQ_API_KEY 환경변수가 설정되어 있지 않음")

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
                timeout=timeout,
            )
        except requests.RequestException as e:
            raise GroqError(f"Groq 요청 실패: {e}") from e

        if resp.status_code == 429:
            wait_s = _parse_retry_after(resp) + 0.5
            last_err = f"rate_limited: {resp.text[:300]}"
            time.sleep(min(wait_s, 60))
            continue

        try:
            resp.raise_for_status()
        except requests.RequestException as e:
            raise GroqError(f"Groq 요청 실패: {e}") from e

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise GroqError(f"Groq 응답 형식 이상: {data}") from e

    raise GroqError(f"재시도 {MAX_RETRIES}회 후에도 rate limit: {last_err}")
