# README_BG — Belief-Graph 기능 요약 및 G0.5 Fine-tuning 가이드

## 1. 무엇을 구현했나

- **목표**: BEHAVIOR Challenge 2026에서 π0.5→G0.5 백본으로, belief graph(기호적 장면 기억)를
  ① conditioning 입력 + ② CoT prediction 타깃 양쪽에 결합
- **belief 규칙** (전 구간 동일 상수): BDDL `:init` prior 0.5 · 관측 0.97/0.03 · 잠정 효과 0.7 ·
  같은 컨테이너 교란 ×0.8(하한 0.6) · **시간 감쇠 없음**

### 구성 요소

- **bgdata (오프라인 파이프라인, `src/bgdata/`)**: raw HDF5 상태 디코딩 → predicate 라벨(10 Hz) → 연산자
  라이브러리(pre≥95%/eff≥85%) → goal COUNT 라인 + `:init` grounding → belief trace →
  CoT 타깃(`cot_targets.parquet`, 1 Hz)
- **빌더 6종** (`src/g05/data_processor/processor/samples_builder.py`)
  - `BeliefGraphBuilder` — conditioning만: `BeliefGraph: <bg_known>` (masked)
  - `+DeltaCoT` / `+UpdateCoT`(Belief:) / `+EffectCoT` / `+SubtaskCoT` — BG 입력 + CoT 타깃
  - `+ObserveCoT` — **model-as-estimator**: 보이는 predicate만 출력, bg_known 입력 없음(지각 순수성)
- **데이터셋 디코딩** (`lerobot_dataset_v3.py`): `bg_*_index` 6열 → tasks 테이블 문자열 조회
- **belief_graph 런타임** (`src/g05/belief_graph/`, torch 무의존)
  - `Belief`(메모리 규칙) · `OperatorLibrary`(효과 grounding·전제조건 검사·Subtask 역파싱) ·
    `GoalSpec`(Δ 계산) · `OracleEstimator`(로컬 전용) · `parse_observe`
  - `BeliefGraphRuntime`: step() → `bg_known` / on_model_cot() → Observe 주입·Delta 교차검증(불일치 시
    conf ×0.8)·Subtask 전제조건 거부 / on_skill_complete() → 효과 적용
- **serving glue** (`src/g05/belief_graph/serving/`): `TaskRegistry` · `SkillController`(관측 확인
  기반 완료/타임아웃) · `BeliefGraphMiddleware`(before_infer/after_infer 2줄 배선) · replay_harness
  - 서버 접점: `build_obs_dict()`의 `plan` 패턴(주입) + `_cot_text` pop 패턴(회수)
- **도구** (`tools/`): `build_bg_fields.py`(타깃→tasks 테이블 확장) · `join_bg_dataset.py`(Stage-3
  join, 1 Hz→30 fps forward-fill) · `eval_estimator.py`(estimator 공통 게이트) · `dryrun_bg_batch.py`

### 검증 완료 (task 45 · 200 에피소드 · SNU-104 CPU/docker)

- 빌더/런타임 smoke 전부 PASS · oracle/observe-gt 하니스 **전 predicate 1.0**
- replay harness(glue end-to-end) **value agreement 0.9923** (잔차 = 효과 적용 시점 차이만)
- dry-run: join(444 문자열·1.83M행) → 디코딩 → 빌더 5/5 매칭 → 최종 문자열·마스크·길이 확인

## 2. G0.5 결합 후 Fine-tuning 가이드

### 선결 조건 (순서대로)

1. **체크포인트**: `huggingface.co/OpenGalaxea/G05`에서 `g05-base` + processor를
   `checkpoints/qwen3_5_2b_base_processor` 경로로 배치 (repo에는 미포함, `.gitignore`됨)
2. **bgdata 산출물**: `PYTHONPATH=src python -m bgdata.run --task N` + `cot_targets`
   - 데이터 루트는 `BGDATA_ROOT` 환경변수 또는 CWD(b1k_raw/·b1k_demos/·BEHAVIOR-1K-main/ 위치)
   - task 45는 SNU-104 `~/b1k_workspace/out/task045`에 생성 완료
3. **Stage-3 join**: `tools/build_bg_fields.py` → `tools/join_bg_dataset.py`
   - `--index-offset`은 기존 tasks 테이블 max(task_index)+1 이상 (join이 충돌 검사함)
   - scratch 데이터셋 방식 권장(원본 불변) — task 45 scratch는 `~/b1k_workspace/scratch_bg_dataset`
4. **dry-run 게이트 (GPU 투입 전 필수)**: `bash bg-g05-merge/dryrun_snu104.sh --scan 300`
   - 확인 항목: bg 필드 디코딩 / 빌더 5/5 / 최종 문자열·loss 마스크 / **토큰 길이**(체크포인트
     배치 후 자동 측정 — 현재 문자 수 프록시: 최대 FULL 651자·TARGET 499자)

### 학습 설정

- 손실 변경 없음 — CoT도 "그냥 토큰", generative segment의 next-token CE 하나
- `MixedSamplesBuilder` 후보 가중치 (샘플당 1개 포맷, G0.5 레시피):

```yaml
processor:
  samples_builder:
    _target_: g05.data_processor.processor.samples_builder.MixedSamplesBuilder
    _partial_: true
    _recursive_: false
    candidates:
      - {_target_: ...samples_builder.BeliefGraphSubtaskCoTBuilder, weight: 3.0}
      - {_target_: ...samples_builder.BeliefGraphDeltaCoTBuilder,   weight: 2.0}
      - {_target_: ...samples_builder.BeliefGraphUpdateCoTBuilder,  weight: 2.0}
      - {_target_: ...samples_builder.BeliefGraphObserveCoTBuilder, weight: 2.0}
      - {_target_: ...samples_builder.BeliefGraphEffectCoTBuilder,  weight: 1.0}
    eval_builder:
      _target_: ...samples_builder.BeliefGraphDeltaCoTBuilder
```

- **robustness**: conditioning 노이즈(dropout/flip) 미구현 상태 — BG 없는 빌더
  (`SubtaskCoTBuilder` 등)를 후보에 소량 섞어 "BG 부재" dropout 근사 권장
- **Observe 토큰 예산**: 5종 중 최장(≤499자) — 토크나이저 실측 후 초과 시
  `bgdata/cot_targets.py MAX_OBSERVE` 축소 또는 참-값만 서술로 압축
- 행동 매핑: R1 Pro 23-d → 통합 27-d = left 7+pad2 | grip 1 | right 7+pad2 | grip 1 |
  lower_body 7(base 3+torso 4, 패딩 없음)

### 학습 후 검증 (오프라인, GPU 추론만 필요)

- `Observe:` 출력 → `tools/eval_estimator.py`(CoTObserveEstimator)로 truth 대비
  precision/recall — **oracle=1.0이 상한**, 이 수치로 `model_obs_conf`(기본 0.9) 보정
- `Delta:`/`Belief:` 예측 vs 외부 belief 일치율 — 기억(mem) 구간 분리 리포트가 핵심
- ladder 비교: Oracle → Observe-CoT(model-as-estimator) → (필요 시) RGB-D hybrid

### 추론 연결 (serving)

```python
mw = BeliefGraphMiddleware(TaskRegistry(bg_artifacts_dir), estimator=None)  # None=model-as-estimator
mw.reset(task_id=45, task_text=raw_obs["task"])
data["bg_known"] = mw.before_infer(raw_obs)   # build_obs_dict 직후
...
mw.after_infer(action.pop("_cot_text", None)) # AR 디코딩 직후
```

- skill 경계 턴만 CoT on(Subtask+Delta), action 턴은 no-CoT(지연 최소화)
- 거부 시 `requery_prompt`로 최대 3회 재질의 · 완료는 controller가 관측으로 판정

### 알려진 한계

- 학습 타깃 = GT-마스킹 belief trace(학습된 추정기 아님) · task 45 단일 태스크 검증
- OmniGibson 시뮬 통합(glue P3)·replay 기반 GT seg는 Linux+NVIDIA 필요
- 상세: `README.md`(병합 가이드) · `E2E_GUIDE.md`(전 과정) · `HYBRID_AND_GLUE_PLAN.md`(잔여 계획)
