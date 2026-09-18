# HYBRID_AND_GLUE_PLAN — 남은 두 모듈의 정의와 구현 계획

> **업데이트 (2026-09-17)**: estimator에는 이 문서의 지각 스택(hybrid) 외에
> **model-as-estimator 대안이 구현되어 병합에 포함**되었습니다 —
> `BeliefGraphObserveCoTBuilder`(학습: visible predicate를 `Observe:` CoT로 출력) +
> `parse_observe`/`on_model_cot`(추론: 그 출력을 외부 belief에 관측으로 주입, `model_obs_conf`
> 보정). 무감쇠 기억·교란 규칙·Delta 교차검증은 외부에 유지됩니다. 아래 hybrid 계획의
> **P0 하니스는 두 경로 모두의 평가 게이트**로 그대로 유효하며, ladder 실험
> (Oracle → Observe-CoT → hybrid → 완전 내재화)으로 비교합니다. hybrid P1–P3의 착수는
> Observe-CoT의 P0 정확도를 본 뒤 결정해도 됩니다.

bg-g05-merge의 알려진 한계 두 가지 — **RGB-D hybrid estimator**와 **서빙 glue** — 가 무엇이고,
어떤 순서로 구현할지 정리한다. 두 모듈이 완성되면 Oracle(특권) 없이 평가 규정
(RGB + depth + proprio + BDDL만 허용) 안에서 belief graph 경로가 완결된다.

---

## 1. RGB-D hybrid estimator — 무엇인가

`BeliefGraphRuntime`은 매 belief 스텝마다 `estimator.estimate(obs, step) → {key: Estimate(value, prob, source)}`
를 호출한다. 지금은 `OracleEstimator`가 특권 truth 파일에서 이 dict를 만들지만(로컬 테스트 전용),
평가에서는 **온보드 관측만으로** 같은 계약을 채워야 한다. "hybrid"는 predicate 갈래별로 추정
방식이 다르다는 뜻이다:

| 갈래 | predicate | 추정 방식 | 이미 확보된 것 |
|---|---|---|---|
| geom | inside / ontop / onfloor | 검출→depth 역투영으로 얻은 물체 3D 위치·AABB에 **라벨과 동일한 기하 규칙** 적용 | 규칙과 임계값 전부 (`bgdata/labels.py`, `config_fitted.json`) |
| appear | open / toggled_on / cooked | 검출 crop → **학습된 분류기** (상태별 binary head) | crop 라벨의 원천 (`predicates.parquet`의 value+visible) |
| robot | inhand / reachable / visited | proprio(EEF pose·gripper) + 추정된 물체 위치에 fitted 규칙 | d_grasp 0.068 m · gripper 0.079 · d_reach 1.50 m (fitted) |

핵심 계약: **이번 스텝에 실제로 본 predicate만 반환**한다(비관측은 반환하지 않음 → belief가
무감쇠 기억으로 유지). 즉 학습 라벨의 `visible` 플래그가 하던 역할을 "검출 성공 여부"가 대신한다.

### 구성 요소

```
src/g05/belief_graph/hybrid/
├── detector.py      # open-vocab 검출+추적: BDDL synset→category 어휘로 질의, 3캠 RGB,
│                    #   인스턴스 트랙 id 부여 (hotdog_a/b — COUNT 설계라 정체성 스왑 허용)
├── localize.py      # mask×depth 역투영 → base frame 3D 중심/AABB → odometry로 지역 좌표계 유지
├── geom_rules.py    # labels.py의 inside/ontop/onfloor 규칙 포트 (입력만 GT→추정 위치로 교체)
├── appear_clf.py    # crop 분류기 (open/toggled_on/cooked) + 온도 보정(calibration)
├── hybrid_estimator.py  # EstimatorProtocol 구현: 세 갈래 병합, prob 보정, 스텝 캐시
└── grounding.py     # 검출 카테고리 ↔ BDDL synset 인스턴스 counts (bgdata/naming.py 브리지 재사용)
```

### 구현 단계

**P0 — 오프라인 벤치마크 하니스 (선행, 저비용, 가장 중요)**
- 데모에는 RGB-D+proprio와 truth가 **둘 다** 있으므로, 시뮬레이터 없이 estimator를 정량 평가할
  수 있다: `tools/eval_estimator.py` — 에피소드를 프레임 순회하며 estimator 출력 vs
  `truth_*.json`(visible=True 부분집합) 비교 → predicate별 precision/recall/보정 곡선.
- 완료 기준: OracleEstimator가 이 하니스에서 100 %가 나오는지로 하니스 자체를 검증.
- 이 하니스가 이후 모든 단계의 게이트가 된다.

**P1 — 관측 기하 확정 (열린 질문 2개 해소)**
- depth 비디오(gray12le)의 스케일 단위 검증, 카메라 intrinsics 확보(설정/캘리브레이션 파일 또는
  OmniGibson VisionSensor 파라미터). → `localize.py`의 역투영이 GT pose 대비 오차 < 수 cm인지
  demo에서 확인 (raw 디코딩 pose가 정답지 역할).
- odometry: proprio `base_qvel` 적분 + (가능하면) 시각 보정. reachable/visited가 요구하는
  정확도는 낮음(임계 1.5 m).

**P2 — geom + robot 갈래 (학습 없음)**
- detector는 우선 기성 open-vocab 모델로, 어휘는 태스크 BDDL의 category 목록.
- `geom_rules.py`·robot 규칙 포트 → P0 하니스에서 평가. 목표: visible 구간에서
  inside/ontop/inhand recall ≥ 0.9 (규칙·임계값이 라벨과 동일하므로 병목은 검출·역투영 품질).

**P3 — appear 분류기**
- crop 데이터셋: 1차는 detector 박스 × `predicates.parquet` 라벨(값·visible)로 약라벨 구축;
  2차는 OmniGibson replay(Linux+GPU)에서 GT seg bbox로 교체 — replay는 perturbation 데이터
  생성과 같은 배치로 계획(DATA_PIPELINE_DESIGN §7).
- 작은 backbone(frozen 특징 + linear head 수준부터) → 온도 보정으로 `prob`가 Known 텍스트의
  confidence로 의미를 갖게 함. 목표: open/toggled_on AUROC > 0.95, cooked는 관측 가능 구간 한정.
- cooked의 근본 한계(닫힌 용기 안 완성)는 estimator가 아니라 process-model TODO임을 유지.

**P4 — 통합·지연 예산**
- `HybridEstimator`로 세 갈래 병합, `belief_every=4` 주기에 맞춘 파이프라이닝
  (무거운 검출은 더 낮은 주기 + 트랙 보간), 평가 컨테이너 패키징.
- 최종 게이트: P0 하니스 전 에피소드 + 시뮬 스모크에서 belief trace와의 Known 일치율 리포트.

---

## 2. 서빙 glue — 무엇인가

빌더·런타임은 계약만 정의한다: 매 control step에 `data["bg_known"]`이 채워져 있어야
eval_builder가 매칭되고, AR 출력의 CoT 텍스트가 `on_model_cot()`로, skill 완료가
`on_skill_complete()`로 되돌아와야 belief가 갱신된다. **이 왕복을 GalaxeaVLA 정책 서버의
요청 처리 경로에 실제로 배선하는 코드**가 glue다. 챌린지 평가는 OmniGibson 평가 루프가
websocket으로 정책 서버를 호출하는 구조이므로, glue의 자연스러운 위치는 서버의 세션/추론 핸들러다.

### 구성 요소

```
src/g05/belief_graph/serving/
├── task_registry.py   # task 이름/id → (operators_*.json, goal_*.json, :init prior 라인) 로딩.
│                      #   :init grounding은 detector 인스턴스 id 기준 (naming.py 브리지 + COUNT)
├── controller.py      # 최소 skill 완료 판정: 현재 skill의 양(+) effect가 belief에서 '관측으로'
│                      #   충족되면 완료, 스텝 타임아웃 시 실패 → 재질의 (bg_pi05 Controller 축약판)
├── policy_middleware.py # 핵심 glue: 서버 세션에 BeliefGraphRuntime 1개 보유
│                      #   reset(task) → registry 조회 + runtime.reset()
│                      #   infer(obs) → runtime.step() → data["bg_known"] 주입 → 내부 정책 호출
│                      #   → AR CoT 텍스트 추출 → runtime.on_model_cot() → controller.tick()
│                      #   → 완료 시 runtime.on_skill_complete() → action 반환 + bg 로그(jsonl)
└── replay_harness.py  # 데모 obs + OracleEstimator로 glue 전체를 오프라인 재생하는 테스트
```

### 구현 단계

**P0 — 서버 접점 조사**
- GalaxeaVLA 서빙 스택에서 (a) 세션 수명(에피소드 reset 신호), (b) 추론 시 processor에 넘어가는
  data dict의 조립 지점, (c) AR 디코딩 텍스트를 응답에서 꺼낼 수 있는 지점을 특정.
  (`utils/websocket`, `tests/test_serve_policy_dynamic_batching.py`, `scripts/` 참조.)
- 완료 기준: "이 함수의 이 dict에 bg_known을 넣으면 eval_builder가 매칭된다"는 위치 1곳 확정.

**P1 — middleware + registry (Oracle로 end-to-end)**
- `policy_middleware.py`·`task_registry.py` 구현. estimator는 Oracle로 두고
  `replay_harness.py`로 데모 재생: 기존 `belief_trace_*.jsonl`과 bg_known 시계열이 일치해야 함
  (기대 불일치는 효과 적용 시점 — annotation 경계 vs controller 판정 — 뿐이며 이를 리포트).
- CoT 파싱 규약 확정: eval_builder가 Delta만 낼 때 vs prompt 템플릿으로 Subtask+Delta를 켤 때의
  두 모드(액션 턴은 no-CoT — G0.5 평가 기본).

**P2 — controller와 거부 루프**
- `controller.py`: 완료 판정(양의 effect가 관측으로 충족·N스텝 유지) + 타임아웃.
- 전제조건 hard 거부 시 재질의: 거부 사유를 bg_known 뒤에 덧붙여 CoT 턴 1회 재실행(최대 3회,
  bg_pi05 규약). 실패 predicate 감지(onfloor 등)는 hybrid P2 이후에만 유의미.

**P3 — 시뮬 통합 (Linux+GPU)**
- OmniGibson 평가 루프 ↔ websocket 서버 스모크: task 45, Oracle→Hybrid 순서로 교체.
- 최종 게이트: 특권 정보 접근이 코드 경로에 없음을 감사(OracleEstimator import 금지 lint),
  episode당 bg 로그(jsonl)로 사후 분석 가능.

---

## 의존 관계와 순서

```
hybrid P0(하니스) ──→ P1(기하) ──→ P2(geom/robot) ──→ P3(appear) ──→ P4(통합)
glue   P0(접점)   ──→ P1(middleware, Oracle) ──→ P2(controller) ──┐
                                                                  └→ P3(시뮬 통합, hybrid 교체)
```

- 두 트랙은 **병렬 진행 가능** — 접합점은 glue P3에서 estimator만 교체하는 것.
- 가장 먼저 할 일 두 가지: hybrid P0의 오프라인 하니스(모든 후속 단계의 게이트),
  glue P0의 서버 접점 조사(설계 확정에 필요한 유일한 미지수).
- 외부 의존/열린 질문: depth 스케일·intrinsics(hybrid P1), replay 가능한 Linux+GPU 머신
  (hybrid P3의 GT crop, glue P3), challenge 데이터의 G0.5 사전학습 포함 여부(ActionCodec)는
  이 두 모듈과 무관하게 별도.
```
