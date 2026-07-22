# ai-security-dataset-worker

Ai-Security 프로젝트(방어적 보안 파인튜닝 데이터셋 구축)의 instruction rationale 다양화를
사용자 로컬 PC 없이 GitHub Actions + Groq 무료 API로 처리하기 위한 워커 저장소.

- `dataset/instruction/evidence_grounded_seed/records.jsonl`: 처리 대상 레코드(학습 매니페스트 갭 중
  아직 다양화 안 된 것)만 담은 부분집합. 소스는 이미 허용형 라이선스(public domain / MIT / Apache /
  CISA 정부저작물 등)로 확인된 것만 사용.
- `dataset/instruction/evidence_grounded_seed/evidence_index.json`: CWE/CAPEC/ATT&CK/D3FEND 등
  근거 ID의 짧은 정의 스니펫(각 350자 이하) 미리 계산본. 환각(모델이 ID 의미를 잘못 추측) 방지용.
- `.github/workflows/generate.yml`: 6시간마다(또는 수동 실행) Groq 무료 티어로 배치 처리 후
  결과를 저장소에 커밋. `--resume`이라 중단/재개 안전.

결과(`records_reasoning.jsonl`)는 본 저장소 원본 프로젝트(로컬)의 병합 스크립트로 주기적으로
가져가 `records.jsonl`에 병합한다.
