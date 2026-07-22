"""공통 IO 유틸: 경로 관리, JSON/JSONL 읽고 쓰기, 다운로드 캐시."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# 프로젝트 루트 = 이 파일 기준 3단계 위 (src/common/io.py -> 루트)
ROOT = Path(__file__).resolve().parents[2]

SCHEMA_DIR = ROOT / "schema"
DATASET_DIR = ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"
NORMALIZED_DIR = DATASET_DIR / "normalized"
GRAPH_DIR = ROOT / "graph"
LOGS_DIR = ROOT / "logs"


def now_iso() -> str:
    """현재 시각 ISO8601 UTC 문자열."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    """레코드들을 JSONL로 저장하고 개수를 반환."""
    ensure_dir(path.parent)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
