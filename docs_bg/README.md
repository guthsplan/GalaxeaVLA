# bg-g05-merge — Belief-Graph builders for GalaxeaVLA (G0.5)

BEHAVIOR Challenge 2026 belief-graph 파이프라인(`bgdata`)의 산출물을 G0.5 학습/추론에 연결하는
변경분입니다. BG(belief graph)를 **conditioning 입력**(`BeliefGraph:` 슬롯, masked)과
**CoT prediction 타깃**(`Delta:` / `Belief:` / `Effect:` / `Subtask:`) 양쪽에 넣습니다
(설계 근거: 워크스페이스의 `G05_REDESIGN.md`).

파일은 GalaxeaVLA repo-상대 경로 그대로 배치되어 있습니다. **git 저장소가 아닌 스냅숏에서
수정했으므로 diff 파일이 없습니다** — 파일 전체 복사로 병합하되, upstream이 이 스냅숏과
다르면 아래 "정확한 변경 hunk"만 수동 적용하면 됩니다.

## 포함 파일

| 파일 | 종류 | 내용 |
|---|---|---|
| `src/g05/data_processor/processor/samples_builder.py` | 수정 (전부 additive) | `BeliefGraphBuilder` + 파생 4종, 모듈 docstring 1줄, smoke test 케이스/검증 블록 |
| `src/g05/data/lerobot/lerobot_dataset_v3.py` | 수정 (hunk 3개) | `bg_*_index` → tasks 테이블 디코딩 + in-memory 컬럼 화이트리스트 등록 |
| `src/g05/belief_graph/` (6개 파일) | **신규 패키지** | **inference용 외부 belief 모듈** — belief 메모리 규칙·연산자 라이브러리·goal Δ 계산·estimator 인터페이스·per-step 런타임 (torch 무의존) |
| `src/bgdata/` (12개 파일) | **신규 패키지** | **오프라인 데이터 파이프라인 전체**(라벨→연산자→goal→trace→CoT 타깃) — 이제 repo에 포함, 데이터 루트는 `BGDATA_ROOT` env 또는 CWD |
| `src/g05/belief_graph/serving/` (5개 파일) | **신규 (glue P0–P2)** | `TaskRegistry`(bgdata 산출물 로딩, goal json의 grounded `init_lines` 포함) · `SkillController`(관측 기반 완료/타임아웃) · `BeliefGraphMiddleware`(`before_infer`/`after_infer` 2줄 배선 — 접점: `build_obs_dict()`의 plan 패턴 + `_cot_text` 회수 패턴) · `replay_harness`(데모 오프라인 재생 검증) |
| `tools/eval_estimator.py` | **신규 (hybrid P0 하니스)** | 모든 estimator의 공통 게이트: truth 대비 predicate별 precision/recall/값 정확도. oracle·observe-gt 내장(둘 다 1.0 필수) |
| `tools/build_bg_fields.py` | 신규 | bgdata `cot_targets.parquet` → tasks 테이블 확장(`bg_tasks.jsonl`) + 프레임별 index parquet 변환기 |
| `deploy_snu104.sh` | 신규 | SNU-104 병합 + **GPU 불필요 범위 전체**(bgdata 재생성 → 필드 변환 → 검증 배터리 5종)를 원샷 실행 |

## 추가된 빌더

| 클래스 | template 핵심 | required_fields / eval_required_fields |
|---|---|---|
| `BeliefGraphBuilder` | `… Task: <command> BeliefGraph: <bg_known_text_!> State: …; Action: <EOV><EOC><action_action>\|` | `bg_known` / `bg_known` |
| `BeliefGraphDeltaCoTBuilder` | + `<EOC><bg_delta_text>\|` — 잔여 goal COUNT 라인 (`Delta:`) | `bg_known, bg_delta` / `bg_known` |
| `BeliefGraphUpdateCoTBuilder` | + `<EOC><bg_belief_text>\|` — 갱신 belief 요약 (`Belief:`; MemoryCoT의 prev/update 구조와 동형) | `bg_known, bg_belief` / `bg_known` |
| `BeliefGraphEffectCoTBuilder` | + `<EOC><bg_effect_text>\|` — 연산자 효과 (`Effect: (open fridge) 1>0`) | `bg_known, bg_effect` / `bg_known` |
| `BeliefGraphSubtaskCoTBuilder` | + `<EOC><atomic_task_text>\|` — BG conditioning + 기존 `Subtask:` 타깃 | `bg_known, atomic_task` / `bg_known` |
| `BeliefGraphObserveCoTBuilder` | `<EOC><bg_observe_text>\|` — **model-as-estimator**: 이번 프레임에 보이는 predicate를 값과 함께 출력 (`(open fridge) 1 \| …`). 의도적으로 `bg_known` conditioning 없음 — 관측이 기억이 아닌 이미지에서 나오도록 | `bg_observe` / (없음) |

설계 노트:
- `bg_known`은 직렬화 텍스트와 JSON(`{"remaining":[…], "known":[[key,p,obs],…]}`) 모두 수용
  (BBoxCoTBuilder의 JSON 패턴 준용). CoT 타깃의 `Delta:/Belief:/Effect:` 접두사는 멱등 처리.
- `eval_required_fields=("bg_known",)` — 추론 시 외부 belief 모듈이 conditioning만 공급하면
  CoT는 모델이 AR로 생성 (repo의 eval 매칭 규약 그대로).
- 학습 샘플링은 기존 `MixedSamplesBuilder`로 가중 선택 (아래 config 예).

## `lerobot_dataset_v3.py` 정확한 변경 hunk (수동 적용용)

**hunk 1 — `load_hf_dataset()`의 `_meta` set** (기존 `'bbox_index', 'action_hint_index', '2d_trace_index',` 줄 뒤):
```python
                'bg_known_index', 'bg_belief_index', 'bg_delta_index', 'bg_effect_index',
```

**hunk 2 — `_materialize_numpy()`의 `_META_COLS` set** (기존 `'bbox_index', 'action_hint_index',` 줄 뒤):
```python
            'bg_known_index', 'bg_belief_index', 'bg_delta_index', 'bg_effect_index',
```

**hunk 3 — `__getitem__` 디코딩 블록** (`high_level_instruction` 디코딩과 `quality_index` 주석 사이):
```python
        # to support belief-graph conditioning/CoT (BEHAVIOR bgdata pipeline):
        #   bg_known_index  -> item["bg_known"]  serialized Remaining/Known text or JSON (input)
        #   bg_belief_index -> item["bg_belief"] belief summary CoT target
        #   bg_delta_index  -> item["bg_delta"]  remaining goal count lines CoT target
        #   bg_effect_index -> item["bg_effect"] grounded operator-effect CoT target
        for _bg_field in ("bg_known", "bg_belief", "bg_delta", "bg_effect"):
            _bg_index_key = f"{_bg_field}_index"
            if _bg_index_key in item and item[_bg_index_key] is not None:
                import pandas as _pd
                _bg_idx_raw = item[_bg_index_key]
                if hasattr(_bg_idx_raw, "item"):
                    _bg_idx_raw = _bg_idx_raw.item()
                if _bg_idx_raw is not None and not (
                    isinstance(_bg_idx_raw, float) and _pd.isna(_bg_idx_raw)
                ):
                    filtered = self.meta.tasks[self.meta.tasks["task_index"] == int(_bg_idx_raw)]
                    item[_bg_field] = filtered.index[0] if len(filtered) > 0 else None
```

`samples_builder.py`는 추가만 있습니다: docstring 목록 2줄, `PlanStepCoTBuilder`와
`MixedSamplesBuilder` 사이의 빌더 5개, smoke test의 `CASES` 5항목과 말미 "BeliefGraph Builders"
검증 블록. upstream과 충돌하면 그 블록들을 그대로 이식하면 됩니다.

## inference용 외부 belief 모듈 — `src/g05/belief_graph/`

빌더의 `bg_known` 입력을 **온라인으로** 만드는 쪽입니다 (bgdata `BeliefSim`/`bg_pi05` 스캐폴드의
검증된 규칙을 torch 무의존으로 포팅; 상수 동일 — prior 0.5 · obs 0.97/0.03 · provisional 0.7 ·
disturb ×0.8 floor 0.6 · **시간 감쇠 없음**).

| 파일 | 내용 |
|---|---|
| `belief.py` | `Belief` 테이블 + 규칙 전부 (관측 우선, reachable 배타, 같은 컨테이너 교란, goal-우선 `known_items`) |
| `operators.py` | `OperatorLibrary.from_json` (bgdata `operators_task*.json` 스키마), 효과 grounding, **전제조건 검사**(관측 모순=hard 거부/기억 모순=soft 경고), Subtask 텍스트 역파싱 |
| `goal.py` | `GoalSpec.from_bgdata_json` (bgdata `goal_task*.json`) + `compute_delta` (COUNT 라인·progress) |
| `estimator.py` | `Estimate` 타입, `EstimatorProtocol`, `OracleEstimator`(bgdata `truth_*.json` — **로컬 테스트 전용, 제출 금지**) |
| `runtime.py` | `BeliefGraphRuntime`: per-step 갱신(belief_every) → `bg_known` 직렬화, `on_model_cot()`(**Observe: 파싱 → belief에 관측 주입(model-as-estimator, `model_obs_conf` 보정)**, Subtask 추적+전제조건 검사, **Delta 교차검증** — 불일치 시 goal predicate conf ×0.8, 모델 예측을 belief에 직접 쓰지 않음), `on_skill_complete()`(효과 적용). `estimator=None`이면 순수 model-as-estimator 모드 |

서빙 연결 (control step마다, processor/builder 실행 **전**):

```python
from g05.belief_graph import BeliefGraphRuntime, OperatorLibrary, GoalSpec, OracleEstimator

runtime = BeliefGraphRuntime(
    lib=OperatorLibrary.from_json("operators_task045.json"),
    goal=GoalSpec.from_bgdata_json("goal_task045.json", task="cook hot dogs",
                                   init_lines=[("(inside hotdog_207 fridge_dszchb_0)", True), ...]),
    estimator=OracleEstimator("truth_450010.json"),  # 실평가: RGB-D hybrid estimator로 교체
)
out = runtime.step(obs, step)
data["bg_known"] = out["bg_known"]      # → BeliefGraph* 빌더 eval 경로
# ... AR 디코딩 후:
runtime.on_model_cot(cot_text)          # Subtask/Delta 처리
runtime.on_skill_complete()             # controller가 완료 감지했을 때
```

`bg_known` 직렬화는 학습 텍스트(`tools/build_bg_fields.py`)와 **문자 단위로 동일 포맷**
(`Remaining: … | Known: <key> <p:.2f> obs|mem | …`) — train/inference prompt 정합이 유지됩니다.
(이번 병합에서 빌더 JSON 경로의 `unobs`도 `mem`으로 통일했습니다.)

## 검증 (macOS · python 3.11 · torch 2.x CPU에서 통과 확인)

```bash
cd GalaxeaVLA
PYTHONPATH=src python -m g05.data_processor.processor.samples_builder
# → 기존 케이스 전부 + 신규 4블록 "✓ … All cases OK."
PYTHONPATH=src python -m g05.belief_graph.runtime   # torch 불필요
# → prior 0.5 노출 · 관측 진입 · 무감쇠 기억 · 효과 0.7 · 교란 ×0.8 · Delta 교차검증 · 직렬화 포맷 검사
```

실데이터 통합도 확인: task 45 `truth_450010.json`을 `OracleEstimator`로 재생 시 fr 0–300에서
prior `0.50 mem`, fr 600 첫 관측 `0.03 obs`, fr 6600 기억 유지 `0.03 mem` — bgdata belief trace와
일치. 전제조건 검사는 로봇이 전자레인지 앞일 때 fridge 대상 pick을 `(reachable ?r)` hard
violation으로 올바르게 거부.

신규 검증 블록: 텍스트 passthrough·JSON 직렬화, 접두사 멱등성, 필드 엄격성
(`bg_known` 없으면 `can_handle=False`), eval 경로(`bg_known`만 검사),
`MixedSamplesBuilder`에서 BG 후보 4개의 게이팅.

## 데이터 준비 (bgdata → GalaxeaVLA)

```bash
python tools/build_bg_fields.py \
  --bg-out <bgdata 출력, 예: out/task045> \
  --out    <출력 폴더> \
  --index-offset 100000   # 기존 tasks 테이블 task_index와 충돌하지 않는 시작값으로 조정
```

- 입력: `cot_targets.parquet` (bgdata PoC, 1 Hz; task 45 기준 200 에피소드 × 61,067 프레임)
- 출력: `bg_fields.parquet`(프레임별 4개 문자열), `bg_tasks.jsonl`(중복 제거 359개 문자열 —
  tasks 테이블 확장 행), `bg_frame_indices.parquet`(프레임별 `bg_*_index` 4열)
- `bg_known`(입력)은 직전 1 Hz 스냅숏의 `Remaining … | Known …`, CoT 타깃은 현재 스냅숏 —
  MemoryCoT의 `prev_memory`/`memory` 분리와 동일한 시간 규약. 에피소드 첫 프레임은 자기
  스냅숏(BDDL prior 0.5)으로 폴백.
- **남은 통합 작업** (데이터셋 레이아웃 의존이라 이 병합분에 미포함):
  `bg_tasks.jsonl`을 대상 데이터셋 tasks 테이블에 병합하고, `bg_frame_indices.parquet`의
  index 열을 frame parquet(data chunk)에 join.

## 학습 config 예 (가중치: G05_REDESIGN §2)

```yaml
processor:
  samples_builder:
    _target_: g05.data_processor.processor.samples_builder.MixedSamplesBuilder
    _partial_: true
    _recursive_: false
    candidates:
      - {_target_: g05.data_processor.processor.samples_builder.BeliefGraphSubtaskCoTBuilder, weight: 3.0}
      - {_target_: g05.data_processor.processor.samples_builder.BeliefGraphDeltaCoTBuilder,   weight: 2.0}
      - {_target_: g05.data_processor.processor.samples_builder.BeliefGraphUpdateCoTBuilder,  weight: 2.0}
      - {_target_: g05.data_processor.processor.samples_builder.BeliefGraphEffectCoTBuilder,  weight: 1.0}
    eval_builder:
      _target_: g05.data_processor.processor.samples_builder.BeliefGraphDeltaCoTBuilder
```

## 알려진 한계

- 학습용 `bg_*` 값은 현재 GT-마스킹된 belief trace 출력(추정기 아님) — bgdata PoC README §5의
  한계를 그대로 승계합니다.
- **평가용 estimator는 두 경로 중 택일** — (a) RGB-D hybrid estimator(별도 지각 스택, 미구현;
  `HYBRID_AND_GLUE_PLAN.md`), (b) **model-as-estimator**: `BeliefGraphObserveCoTBuilder`로 학습된
  모델의 `Observe:` CoT를 `on_model_cot()`/`parse_observe()`가 belief 관측으로 소비 (이번 병합에
  포함). (b)의 `model_obs_conf`(기본 0.9)는 truth 대비 오프라인 보정이 필요하며, ladder 실험
  (Oracle → Observe-CoT → 완전 내재화)으로 정확도를 검증해야 함.
- `runtime.step()`을 서빙 루프에 실제로 호출하는 glue(정책 서버/eval wrapper 수정)는 서빙
  스택 구성에 의존해 미포함 — 위 "서빙 연결" 스니펫이 계약의 전부입니다.
- task 45 데이터로만 end-to-end 검증됨. 빌더/디코딩/런타임 코드는 태스크-불가지론적.
- `--index-offset` 기본값 100000은 예시 — 대상 데이터셋의 실제 tasks 테이블 크기 확인 필요.
