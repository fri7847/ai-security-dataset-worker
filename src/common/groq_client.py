"""Groq 무료 티어(OpenAI 호환 API) 클라이언트.

GitHub Actions 등 사용자 로컬 PC 자원을 쓰지 않는 실행 환경에서 rationale 생성 등에
사용하기 위한 백엔드. API 키는 환경변수 GROQ_API_KEY로 전달한다(코드에 하드코딩 금지).
무료 티어 한도(요청 기준, 2026-07 확인): 14,400 req/day, 30 RPM, 6K TPM — 대량 실행 전
반드시 소규모 배치로 실측(RPM 초과 시 429)한 뒤 워커 수/속도를 정할 것(이 프로젝트에서
LM Studio parallel, Claude CLI workers 모두 동일한 실측 없이는 안 믿는 패턴이 유효했음).
"""
from __future__ import annotations

import os

import requests

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


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
        if resp.status_code == 429:
            raise GroqError(f"rate_limited: {resp.text[:300]}")
        resp.raise_for_status()
    except requests.RequestException as e:
        raise GroqError(f"Groq 요청 실패: {e}") from e

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise GroqError(f"Groq 응답 형식 이상: {data}") from e
