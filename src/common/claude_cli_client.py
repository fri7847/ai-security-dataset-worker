"""Claude Code CLI(`claude -p`)를 서브프로세스로 호출하는 클라이언트.

로컬 LM Studio(`llm_client.py`)와 같은 용도(rationale 생성)이지만 속도가 필요할 때
Claude 사용량을 소모하는 대신 빠르게 처리하기 위한 대체 백엔드. 과거 이 프로젝트에서
Claude API가 "cybersecurity topic" 안전분류기에 대화 자체를 막은 이력이 있어(CLAUDE.md
참고) 대량 실행 전 소규모 배치로 차단율을 반드시 재검증할 것.
"""
from __future__ import annotations

import subprocess


class ClaudeCliError(RuntimeError):
    pass


def chat(system_prompt: str, user_prompt: str, model: str = "haiku", timeout: int = 90) -> str:
    """user_prompt는 stdin으로 전달한다(argv로 넘기면 Windows 커맨드라인 길이 한도를
    넘는 긴 프롬프트에서 WinError 206 "파일 이름이나 확장명이 너무 깁니다"로 실패함)."""
    try:
        result = subprocess.run(
            ["claude", "-p", "--model", model, "--system-prompt", system_prompt],
            input=user_prompt,
            capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="ignore",
        )
    except subprocess.TimeoutExpired as e:
        raise ClaudeCliError(f"timeout after {timeout}s") from e
    if result.returncode != 0:
        raise ClaudeCliError(f"exit {result.returncode}: {result.stderr.strip()[:300]}")
    text = result.stdout.strip()
    if not text:
        raise ClaudeCliError(f"empty stdout; stderr={result.stderr.strip()[:300]}")
    return text
