# E2E_GUIDE — bgdata 오프라인 파이프라인부터 BeliefGraphBuilder 사용까지

bg-g05-merge가 GalaxeaVLA official repo에 완전히 병합되었다는 가정 하에, 데이터 생성부터
학습·추론에서 `BeliefGraphBuilder`가 동작하기까지의 전 과정입니다. 모든 명령·수치는 task 45
(cook hot dogs, 로컬 200 에피소드)에서 실행·검증된 값입니다.

```
[Stage 1  bgdata 오프라인]        [Stage 2  변환]           [Stage 3  데이터셋 통합]
raw HDF5 ─┐                                                tasks 테이블 + bg_tasks.jsonl
demos ────┼→ python -m bgdata.run → cot_targets.parquet →  frame parquet + bg_*_index 열
BDDL ─────┘   (+ cot_targets)      (build_bg_fields.py)         │
   │                                                            ▼
   ├→ operators_task045.json ──────────────┐        [Stage 4  학습]
   ├→ goal_task045.json ───────────────────┤        lerobot_dataset_v3가 bg_* 디코딩
   └→ truth_*.json (로컬 테스트용) ─────────┤        → MixedSamplesBuilder가 BG 빌더 선택
                                           │        → 단일 next-token CE
                                           ▼
                                  [Stage 5  추론]
                                  BeliefGraphRuntime.step() → data["bg_known"]
                                  → BeliefGraph* 빌더 eval 경로 → AR 디코딩
                                  → on_model_cot() / on_skill_complete()
```

---

## Stage 0 — 전제

- 데이터: `b1k_raw/task-XXXX/`(HDF5), `b1k_demos/`(LeRobot v3: meta/data/annotations),
  `BEHAVIOR-1K-main/bddl3/` — bgdata 워크스페이스에서 접근 가능해야 함.
- 환경: bgdata는 python 3.11 + numpy/pandas/pyarrow/h5py/matplotlib (CPU 전용).
  GalaxeaVLA 쪽은 repo 표준 환경(torch 포함). `belief_graph` 런타임 자체는 torch 무의존.

## Stage 1 — bgdata 오프라인 파이프라인 (라벨 → 연산자 → belief trace → CoT 타깃)

```bash
cd <데이터 루트>                     # b1k_raw/, b1k_demos/, BEHAVIOR-1K-main/ 이 있는 곳 (또는 BGDATA_ROOT로 지정)
PYTHONPATH=<GalaxeaVLA>/src python -m bgdata.run --task 45 --out out/task045 --episodes all --fit-episodes 20
PYTHONPATH=<GalaxeaVLA>/src python -c "from bgdata.cot_targets import main; main('out/task045', 45)"
# 원격/컨테이너 실행이 필요하면: bash deploy/run_all.sh --fetch  (deploy/README.md 참조)
```

내부 단계와 산출물 (task 45 실측: 총 151 s, 산출물 ≈2.1 GB):

| 단계 | 산출물 | 이후 어디서 쓰이나 |
|---|---|---|
| raw 상태 벡터 디코딩 + 기하/외관 규칙 라벨링 (10 Hz) | `predicates.parquet` (18.9 M행) | truth/trace의 원천 |
| 연산자 추출 (pre ≥95 % · eff ≥85 %) + 검증 (mismatch 3.4 %) | `operators_task045.json` | **추론**: 효과 적용·전제조건 검사 (`OperatorLibrary.from_json`) |
| BDDL goal grounding (COUNT 라인 + synset→scene 매핑) | `goal_task045.json` | **추론**: `GoalSpec.from_bgdata_json` → Remaining 계산 |
| 프레임별 truth 내보내기 | `truth_<episode>.json` ×200 | **로컬 테스트**: `OracleEstimator` 입력 (특권 — 제출 금지) |
| 참조 belief 시뮬레이션 (prior 0.5·obs 0.97·effect 0.7·disturb ×0.8, 무감쇠) | `belief_trace_*.jsonl` ×200 | cot_targets의 원천 |
| G0.5 CoT 타깃 생성 (1 Hz) | `cot_targets.parquet` (61,067행: subtask/belief/delta/effect) | **Stage 2 입력** |

## Stage 2 — GalaxeaVLA 필드 형식으로 변환

```bash
cd GalaxeaVLA
python tools/build_bg_fields.py \
  --bg-out <bgdata>/out/task045 \
  --out    <출력 폴더> \
  --index-offset <기존 tasks 테이블 마지막 task_index + 1>
```

- `bg_known`(conditioning 입력) = **직전** 1 Hz 스냅숏의 `Remaining … | Known …`
  (MemoryCoT의 prev/update 시간 규약과 동일; 에피소드 첫 프레임은 BDDL prior 스냅숏).
- CoT 타깃(`bg_belief/bg_delta/bg_effect`) = **현재** 스냅숏.
- 산출물: `bg_fields.parquet`(프레임별 문자열 4열, 디버깅용),
  `bg_tasks.jsonl`(중복 제거 문자열 — task 45는 359개),
  `bg_frame_indices.parquet`(episode, frame, `bg_known_index` 등 int 4열).

## Stage 3 — 데이터셋 통합 (tasks 테이블 + frame parquet)

GalaxeaVLA의 텍스트 필드 규약(`<field>_index` int 열 → tasks 테이블 문자열 조회)에 맞춰:

1. `bg_tasks.jsonl`의 행들을 대상 데이터셋의 tasks 테이블에 **추가** (task_index 충돌 없어야 함 —
   `--index-offset`으로 보장).
2. `bg_frame_indices.parquet`를 (episode_index, frame_index) 기준으로 data chunk parquet에
   join해 `bg_known_index / bg_belief_index / bg_delta_index / bg_effect_index` 4열을 추가.
   bgdata 프레임은 30 fps 인덱스의 1 Hz 서브샘플이므로, 중간 프레임은 직전 1 Hz 값으로
   forward-fill (belief는 그 사이 상수라는 가정이 belief trace의 시간 해상도와 일치).
3. 끝 — 병합된 `lerobot_dataset_v3.py`가 이 4열을 자동으로 감지·디코딩해
   `data["bg_known"]` 등 문자열 필드로 만들어 줍니다 (in-memory 모드 화이트리스트 포함).

## Stage 4 — 학습에서 BeliefGraphBuilder 사용

config에서 `MixedSamplesBuilder`로 BG 빌더들을 가중 샘플링 (G05_REDESIGN §2의 가중치):

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
      - {_target_: g05.data_processor.processor.samples_builder.BeliefGraphObserveCoTBuilder, weight: 2.0}
    eval_builder:
      _target_: g05.data_processor.processor.samples_builder.BeliefGraphDeltaCoTBuilder
```

`BeliefGraphObserveCoTBuilder`(model-as-estimator 학습)는 의도적으로 `bg_known` conditioning이
없습니다 — 관측이 기억이 아니라 이미지에서 나오도록 강제하기 위함입니다.

학습 배치마다 일어나는 일: 프레임에 4개 bg 필드가 모두 있으면 후보 4개가 전부
`can_handle=True` → 가중 무작위로 **정확히 1개** 빌더 선택(G0.5 레시피) → 예:
`BeliefGraphDeltaCoTBuilder`가 만드는 토큰 스트림:

```
[user] <images> Embodiment: r1pro; Task: cook hot dogs
       BeliefGraph: Remaining: (cooked ?x) 0/2 [hotdog.n.02] | Known: (inhand hotdog_207) 0.97 obs | …   ← masked (loss 없음)
       State: <proprio>;
[assistant] predict remaining goal predicates
<EOC> Delta: (cooked ?x) 0/2 [hotdog.n.02] |            ← CoT 타깃 (CE)
Action: <EOV> <action codes> | <eos>                     ← 행동 타깃 (CE)
```

손실은 기존 그대로 generative segment의 next-token CE 하나 — BG CoT도 "그냥 토큰"입니다.

## Stage 5 — 추론에서 BeliefGraphRuntime + eval 경로

빌더의 `eval_required_fields=("bg_known",)`이므로, 서빙 루프가 매 control step에
`data["bg_known"]`만 채우면 eval_builder가 매칭되고 CoT는 모델이 AR로 생성합니다:

```python
from g05.belief_graph import BeliefGraphRuntime, OperatorLibrary, GoalSpec, OracleEstimator

runtime = BeliefGraphRuntime(
    lib=OperatorLibrary.from_json("operators_task045.json"),     # Stage 1 산출물
    goal=GoalSpec.from_bgdata_json("goal_task045.json",          # Stage 1 산출물
        task="cook hot dogs",
        init_lines=[("(inside hotdog_207 fridge_dszchb_0)", True),
                    ("(inside hotdog_208 fridge_dszchb_0)", True),
                    ("(cooked hotdog_207)", False), ("(cooked hotdog_208)", False)]),
    estimator=OracleEstimator("truth_450010.json"),  # 로컬 테스트
    # 실평가 옵션 A: estimator=None (model-as-estimator — 모델의 Observe: CoT가 관측 공급,
    #   on_model_cot()가 model_obs_conf로 belief에 주입)
    # 실평가 옵션 B: RGB-D hybrid estimator (HYBRID_AND_GLUE_PLAN.md, 미구현)
    belief_every=4,
)

# ---- control step마다 ----
out = runtime.step(obs, step)          # ① estimator → belief 갱신(무감쇠 메모리) → Δ 계산
data["bg_known"] = out["bg_known"]     # ② eval_builder(BeliefGraphDeltaCoTBuilder) 입력
cot_text, action = model.infer(...)    # ③ AR: "Delta: …" CoT + action codes
hooks = runtime.on_model_cot(cot_text) # ④ Subtask 역파싱→전제조건 검사(관측 모순=거부),
                                       #    Delta 교차검증(불일치→goal predicate conf ×0.8)
if controller_detected_completion:     # ⑤ skill 완료 시
    runtime.on_skill_complete()        #    연산자 효과 0.7 + 같은 컨테이너 교란 ×0.8
```

train/inference 정합의 핵심: `runtime.serialize_known()`이 만드는 `bg_known` 텍스트가 Stage 2의
학습 직렬화와 **문자 단위로 동일 포맷**(`Remaining: … | Known: <key> <p:.2f> obs|mem | …`)입니다.

## 검증 체크리스트 (이 스냅숏에서 통과 확인)

```bash
PYTHONPATH=src python -m g05.data_processor.processor.samples_builder   # 빌더 전체 smoke test
PYTHONPATH=src python -m g05.belief_graph.runtime                       # 규칙 7종 smoke test (torch 불필요)
```

실데이터 교차검증: `OracleEstimator("truth_450010.json")` 재생 시 fr 0–300 prior `0.50 mem` →
fr 600 첫 관측 `0.03 obs` → fr 6600 무감쇠 기억 `0.03 mem` — bgdata `belief_trace_450010.jsonl`과 일치.

## 한계 (승계)

- 학습 bg 값 = GT-마스킹 belief trace 출력 (학습된 추정기 아님) — bgdata README §5.
- 실평가용 RGB-D hybrid estimator와 `runtime.step()`을 호출하는 서빙 glue는 미포함
  (`EstimatorProtocol` 인터페이스와 위 스니펫이 계약의 전부).
- task 45 외 태스크는 Stage 1을 해당 태스크로 재실행해야 하며, bgdata의 객체 로스터
  일반화(`bgdata/naming.py`의 synset 브리지 활용)는 진행 중인 TODO.
