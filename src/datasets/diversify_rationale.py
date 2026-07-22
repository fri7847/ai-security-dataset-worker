"""instruction seed의 `rationale` 필드를 로컬 LM Studio 모델(qwen3-8b)로 다양화한다.

배경: 현재 rationale은 소스별 고정 템플릿 문장이라 다양성이 없음(AGENTS.md 기록됨).
Claude API로 CVE/ATT&CK 등 보안 근거를 설명하는 텍스트를 대량 생성하면 안전분류기에
막히는 경우가 있어, 로컬에 이미 로드된 모델(LM Studio, http://localhost:1234)을 사용한다.

안전장치: 생성된 rationale에 나타난 CVE/CWE/CAPEC/ATT&CK ID가 evidence_ids/answer/prompt에
없는 새 ID면 환각으로 간주하고 원본 템플릿 rationale로 되돌린다(근거 이탈 방지).
재실행 시 이미 처리된 id는 건너뛰어 이어서 진행한다(--resume).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from jsonschema import Draft202012Validator

from ..common.io import ROOT, NORMALIZED_DIR, read_jsonl, ensure_dir, now_iso, write_json
from ..common.llm_client import LmStudioError, chat
from ..common.claude_cli_client import ClaudeCliError, chat as claude_cli_chat
from ..common.groq_client import GroqError, chat as groq_chat

SEED_DIR = ROOT / "dataset" / "instruction" / "evidence_grounded_seed"
DEFAULT_INPUT = SEED_DIR / "records.jsonl"
DEFAULT_OUTPUT = SEED_DIR / "records_reasoning.jsonl"
SCHEMA_PATH = ROOT / "schema" / "instruction.schema.json"

ID_PATTERN = re.compile(
    r"CVE-\d{4}-\d{4,10}"
    r"|CWE-\d{1,4}"
    r"|CAPEC-\d{1,4}"
    r"|D3-[A-Z][A-Za-z]+"
    r"|CAR-\d{4}-\d{2}-\d{3}"
    r"|\bT\d{4}(?:\.\d{3})?\b"
    r"|\bM\d{4}\b"
    r"|\bG\d{4}\b"
    r"|\bS\d{4}\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You write short reasoning explanations for an authorized defensive-security training "
    "dataset. Given a question, its grounded answer, and the evidence record IDs it must rely "
    "on, write 2-3 sentences explaining HOW the answer follows from that evidence. "
    "Rules: (1) use only facts, IDs, and numbers already present in the question/answer/evidence — "
    "never introduce a new CVE, CWE, CAPEC, ATT&CK, or rule ID; (2) never state what a CWE, CVE, "
    "ATT&CK, CAPEC, or D3FEND identifier means from your own background knowledge — only use the "
    "meaning given in the 'Evidence definitions' block below, or in the answer text; if an ID has "
    "no provided definition, treat it as opaque and do not guess what it means; (3) if you lack "
    "extra detail, focus on process — why the cited evidence supports this answer, and what an "
    "analyst should verify next — not on restating or inventing technical meaning; (4) do not "
    "repeat the answer verbatim; (5) no meta-commentary, headers, or markdown, plain sentences only. "
    "/no_think"
)


def extract_ids(text: str) -> set[str]:
    return {m.upper() for m in ID_PATTERN.findall(text)}


def build_evidence_index() -> dict[str, str]:
    """cve 소스(348k, 이미 answer에 본문 포함됨)를 제외한 모든 정규화 소스에서
    id -> 짧은 정의 텍스트 인덱스를 만든다. CWE/CAPEC/ATT&CK/D3FEND 등의 실제
    의미를 프롬프트에 넣어 모델이 배경지식으로 잘못 추측하는 것을 막는다."""
    idx: dict[str, str] = {}
    for src_dir in sorted(NORMALIZED_DIR.iterdir()):
        if not src_dir.is_dir() or src_dir.name == "nvd_cve":
            continue
        f = src_dir / "records.jsonl"
        if not f.exists():
            continue
        for rec in read_jsonl(f):
            snippet = rec.get("text", "")[:350].strip()
            idx[rec["id"]] = f"{rec.get('title', '')} — {snippet}"
    return idx


def build_user_prompt(record: dict, evidence_index: dict[str, str]) -> str:
    context_lines = []
    for eid in record["evidence_ids"]:
        definition = evidence_index.get(eid)
        if definition:
            context_lines.append(f"- {eid}: {definition}")
    context_block = (
        "Evidence definitions (the ONLY source of truth for what these IDs mean):\n"
        + "\n".join(context_lines)
        if context_lines
        else ""
    )
    return (
        f"Task type: {record['task_type']}\n"
        f"Question: {record['prompt']}\n"
        f"Evidence IDs: {', '.join(record['evidence_ids'])}\n"
        f"{context_block}\n"
        f"Grounded answer: {record['answer']}\n\n"
        "Write the reasoning explanation now."
    )


MAX_PROMPT_CHARS = 6000  # LM Studio(로컬, context=8192) 기준 안전 상한. claude_cli/groq 백엔드는 더 큼(아래 참고).
# groq 무료 티어는 TPM(분당 토큰) 6K가 RPM(30)보다 먼저 병목이 됨 — 프롬프트가 길수록
# 동시 요청 처리량이 급감하므로 claude_cli보다 훨씬 낮게 잡음(그래도 대다수 레코드는 통과).
MAX_PROMPT_CHARS_BY_BACKEND = {"lmstudio": 6000, "claude_cli": 60000, "groq": 8000}


def generate_rationale(record: dict, model: str, evidence_index: dict[str, str], backend: str = "lmstudio") -> tuple[str | None, str]:
    """(새 rationale 또는 None, 사유) 반환. None이면 원본 유지."""
    allowed = extract_ids(record["prompt"]) | extract_ids(record["answer"])
    for eid in record["evidence_ids"] + record["source_record_ids"]:
        allowed |= extract_ids(eid)
        definition = evidence_index.get(eid)
        if definition:
            allowed |= extract_ids(definition)

    user_prompt = build_user_prompt(record, evidence_index)
    if len(user_prompt) + len(SYSTEM_PROMPT) > MAX_PROMPT_CHARS_BY_BACKEND.get(backend, MAX_PROMPT_CHARS):
        return None, "prompt_too_long"

    try:
        if backend == "claude_cli":
            text = claude_cli_chat(SYSTEM_PROMPT, user_prompt, model=model)
        elif backend == "groq":
            text = groq_chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                temperature=0.4,
            )
        else:
            text = chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
                temperature=0.4,
            )
    except (LmStudioError, ClaudeCliError, GroqError) as e:
        print(f"[DEBUG API ERROR] {record['id']}: {e}", file=sys.stderr, flush=True)
        return None, f"api_error: {e}"

    # qwen3 계열은 <think> 블록을 낼 수 있어 제거
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if not text:
        return None, "empty_response"

    generated_ids = extract_ids(text)
    unsupported = generated_ids - allowed
    if unsupported:
        return None, f"hallucinated_ids: {sorted(unsupported)}"

    return text, "ok"


def load_done_ids(output_path) -> set[str]:
    if not output_path.exists():
        return set()
    return {r["id"] for r in read_jsonl(output_path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--limit", type=int, default=0, help="0이면 전체")
    ap.add_argument("--task-type", default=None)
    ap.add_argument("--skip-task-type", action="append", default=[], help="반복 지정 가능; 이 task_type들은 제외")
    ap.add_argument("--id-file", default=None,
                    help="한 줄에 id 하나인 파일; 이 id들만 처리(학습 매니페스트에 실제로 쓰이는 레코드만 다양화할 때)")
    ap.add_argument("--evidence-index-file", default=None,
                    help="build_evidence_index()로 미리 만들어둔 {id: definition} JSON 파일 경로. "
                         "지정하면 dataset/normalized 전체를 스캔하지 않고 이 파일을 그대로 씀 "
                         "(원격 실행 환경에 전체 정규화 소스 대신 이 파일 하나만 두면 됨).")
    ap.add_argument("--model", default="qwen/qwen3-8b")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="동시 요청 수(LM Studio는 모델 PARALLEL 설정과 맞출 것)")
    ap.add_argument("--backend", choices=["lmstudio", "claude_cli", "groq"], default="lmstudio",
                    help="lmstudio: 로컬 LM Studio(무료, 느림). claude_cli: `claude -p` 서브프로세스(Claude 사용량 소모, 빠름). "
                         "groq: Groq 무료 API(GROQ_API_KEY 필요, 로컬 자원 불사용, GitHub Actions 등에서 사용) "
                         "— --model도 이 백엔드용 값(예: llama-3.3-70b-versatile)으로 바꿀 것")
    args = ap.parse_args()

    from pathlib import Path
    input_path = Path(args.input)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    records = read_jsonl(input_path)
    if args.task_type:
        records = [r for r in records if r["task_type"] == args.task_type]
    if args.skip_task_type:
        skip = set(args.skip_task_type)
        records = [r for r in records if r["task_type"] not in skip]
    if args.id_file:
        wanted = {ln.strip() for ln in Path(args.id_file).read_text(encoding="utf-8").splitlines() if ln.strip()}
        records = [r for r in records if r["id"] in wanted]
        print(f"id-file filter: {len(wanted)} ids requested, {len(records)} matched", flush=True)

    done_ids = load_done_ids(output_path) if args.resume else set()
    if done_ids:
        records = [r for r in records if r["id"] not in done_ids]

    if args.limit:
        records = records[: args.limit]

    if args.evidence_index_file:
        print(f"loading precomputed evidence index from {args.evidence_index_file}...", flush=True)
        evidence_index = json.loads(Path(args.evidence_index_file).read_text(encoding="utf-8"))
    else:
        print("building evidence definition index...", flush=True)
        evidence_index = build_evidence_index()
    print(f"evidence index: {len(evidence_index)} ids", flush=True)

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    stats = {"generated": 0, "kept_original": 0, "api_error": 0, "hallucinated": 0, "empty": 0, "too_long": 0}
    stats_lock = threading.Lock()
    write_lock = threading.Lock()
    start = time.time()

    def process_one(rec: dict) -> dict:
        new_rationale, reason = generate_rationale(rec, args.model, evidence_index, backend=args.backend)
        out_rec = dict(rec)
        generated = False
        if new_rationale:
            out_rec["rationale"] = new_rationale
            generated = True

        errs = list(validator.iter_errors(out_rec))
        if errs:
            print(f"[SCHEMA INVALID] {rec['id']}: {errs[0].message}", file=sys.stderr)
            out_rec = rec
            generated = False

        with stats_lock:
            if generated:
                stats["generated"] += 1
            else:
                stats["kept_original"] += 1
                if reason.startswith("api_error"):
                    stats["api_error"] += 1
                elif reason.startswith("hallucinated"):
                    stats["hallucinated"] += 1
                elif reason == "empty_response":
                    stats["empty"] += 1
                elif reason == "prompt_too_long":
                    stats["too_long"] += 1
        return out_rec

    mode = "a" if (args.resume and output_path.exists()) else "w"
    with open(output_path, mode, encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, rec) for rec in records]
        for i, fut in enumerate(as_completed(futures), 1):
            out_rec = fut.result()
            with write_lock:
                f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                f.flush()

            if i % 25 == 0 or i == len(records):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed else 0
                print(
                    f"{i}/{len(records)} generated={stats['generated']} "
                    f"kept_original={stats['kept_original']} "
                    f"(hallucinated={stats['hallucinated']} api_error={stats['api_error']} "
                    f"empty={stats['empty']}) rate={rate:.2f}/s",
                    flush=True,
                )

    stats_path = output_path.parent / (output_path.stem + "_stats.json")
    write_json(
        stats_path,
        {
            "generated_at": now_iso(),
            "input": str(input_path),
            "output": str(output_path),
            "model": args.model,
            "processed": len(records),
            **stats,
            "elapsed_seconds": round(time.time() - start, 1),
        },
    )
    print(f"done. stats -> {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
