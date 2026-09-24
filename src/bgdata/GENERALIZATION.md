# bgdata — 100 task 일반화 (2026-09-24)

task 45(`cook_hot_dogs`) 전용이던 belief-graph 라벨 파이프라인을 BEHAVIOR-1K 2026 챌린지 100개
task 전부에서 동작하도록 재작성했다. 하드코딩(객체 이름·임계값·goal)은 모두 제거되었고, 필요한 task
사양은 데이터에서 유도한다.

```
python -m bgdata.run --task N --out out/taskNNN [--episodes K|all] [--fit-episodes 20]
python -c "from bgdata.cot_targets import main; main('out/taskNNN', N)"
python -m bgdata.validate --root out            # task별 품질 리포트
```
래퍼: `scripts/preprocess_bgdata.sh --task all|0-49|"3 7"` (서브모듈 `src/bgdata`를 기본 소스로 사용).

## 1. 입력과 task 사양 유도 (`taskspec.py`)

| 출처 | 용도 |
|---|---|
| raw HDF5 `/data` attrs `config` | `activity_name`, 로봇 이름(`robot_r1` task 0–49 / `robot` 50–99) |
| raw HDF5 attrs `scene_file` | `metadata.task.inst_to_name`(BDDL 인스턴스→scene 인스턴스), `objects_info.init_info`(category/model/scale/fixed_base), `state.registry.object_registry`(초기 pose·joint·non-kinematic 상태 순서) |
| `2026-challenge-task-instances/scenes/<scene>/json/*_template.json` | 녹화 registry에 없는 fixture(countertop, bookcase, floors …)의 pose |
| `behavior-1k-assets/objects/<cat>/<model>/misc/metadata.json` | openable joint 목록 |
| `bddl3/.../object_inventory.json` | 모델별 bbox(×scale = extents) |
| bddl3 `parse_problem(name, 0, "behavior-1k")` + `ObjectTaxonomy` | objects/init/goal, synset ability·param(`cook_temperature` 등) |

`BGDATA_ROOT`(b1k_demos/b1k_raw/bddl3 심링크), `BGDATA_ASSET_META`, `BGDATA_TASK_INSTANCES` 환경변수로
위치를 지정한다(`scripts/preprocess_bgdata.sh`가 `OMNIGIBSON_DATA_PATH` 기준으로 채움).

annotation 토큰은 `resolve_token`으로 인스턴스 후보를 만든다: 인스턴스명 그대로 → category(task 50–99는
category 토큰) → lemma/prefix → fixture. 세그먼트별 실제 인스턴스 결정은 `labels.EpisodeData._resolve_segments`
(조작 대상은 구간 내 최대 이동, 타깃은 구간 끝 최근접). `frame_duration`이 중단·재개로
`[[s1,s2],[e1,e2]]`인 스킬은 part별 세그먼트로 분리한다.

## 2. raw 상태 디코딩 (`rawstate.py`)

OmniGibson dump 포맷을 순차 파싱한다: header 8 floats, registry uuid, 객체 블록
`[uuid, is_asleep, pos3, quat4, lin_vel3, ang_vel3, joint_pos(nj), joint_vel(nj), non_kin…]`
(fixed_base·joint 없는 객체는 velocity 없이 9 floats), non-kin 크기표(Temperature 1, ToggledOn 2,
Saturated 3+2n, ModifiedParticles 1+2n …), 로봇 124 floats, assisted-grasp(AG) 블록 19 floats.
uuid는 `int(float32(md5(name) % 1e8))`로 작고 겹칠 수 있어 **집합 소속으로만** 앵커를 판정한다.
활성 객체만 dump되므로 미dump 프레임은 scene_file 초기 상태에서 carry-forward한다(이때 속도는 0으로
간주 — 수면 객체는 정지 상태).

HDF5에 여러 `demo_N` 그룹(재시도)이 있으면 `action`이 있는 그룹 중 에피소드 길이가 맞는 것을 고른다.

## 3. 라벨 규칙 (`labels.py`, 임계값은 `config.py`)

| predicate | 규칙 |
|---|---|
| `inside a b` | b가 컨테이너(fillable/openable ability **또는** BDDL에서 `inside/filled/contains`의 컨테이너로 쓰임)일 때, a 중심이 b의 AABB(+margin) 안 |
| `ontop a b` | a 바닥이 b AABB 윗면 근처(컨테이너) 또는 b의 수직 범위 안(비컨테이너: 책상 트레이·싱크대 림 등), xy 안, 정지, 손에 없음, b와 같은 컨테이너에 있지 않은 다른 컨테이너 안이 아님 |
| `nextto` | OmniGibson NextTo: AABB gap < mean(dims)/6, 수직 겹침, not inside |
| `under`, `touching`, `onfloor` | AABB 기반 |
| `open` | joint > 5 % range (관측 min/max로 range 피팅) |
| `toggled_on`, `attached`, `cooked/frozen/on_fire` | non-kin 상태(ToggledOn, AttachedTo, Temperature vs synset param) |
| `covered/filled/contains/real` | 입자계는 raw에 없음 → init 값, 성공 종료 시점부터 goal 값 |
| `inhand[_left/_right]` | AG 블록(task 50–99) 또는 EEF 거리·gripper 피팅(0–49) |
| `reachable/visited` | base와 타깃 AABB의 xy 거리 < 피팅값(세그먼트 시작 p95 ×1.15) |
| 가시성 | 카메라 절두체 프록시(AABB 최근접점), 닫힌 컨테이너 안은 비가시 |

**성공 시점 강제**: `terminated`가 켜진 프레임(시뮬레이터가 goal 만족을 판정)부터는 goal이
forall/and로 강제하는 literal(`TaskGoal.forced_literals`)을 goal 값으로 덮어쓴다(exists/or 하위는 제외).
덮어쓴 key 수는 `summary.json.decode_stats[ep].goal_forced_keys`에 남겨 규칙 오차 지표로 쓴다.

key universe(`key_universe`)는 BDDL atom + annotation 토큰 후보의 스킬 관계(pick/place → inside/ontop,
door → open, switch → toggled_on)로 task 단위로 한 번 결정한다.

## 4. goal / operator / belief (`goal.py`, `operators.py`, `belief_trace.py`)

* goal: forall/exists/forn/forpairs/and/or/not 3값 평가, 카운트 라인(`(inside ?x fridge) 1/2 [synset]`)
* operator: 35개 스킬 어휘 → op 템플릿(`SKILL_TABLE`), 세그먼트 전후 라벨 차이에서 pre/eff 통계 추출
* belief: prior 0.5 → 관측 0.97 / 잠정 효과 0.7 / 교란 ×0.8(하한 0.6), 1 Hz `cot_targets.parquet`
  (`Subtask/Belief/Delta/Effect/Observe`, 개체 이름은 모두 scene 인스턴스 id)

## 5. 데이터 현황 (이 서버, `/data0/hoyong/b1k_data`)

* raw HDF5 + annotation: 100 task × 200 에피소드 전부 존재.
* LeRobot 데모 parquet(proprio·카메라 pose): task 0–57 전부, 58(93 %), 61(22 %), **62–99 없음**.
  parquet이 없으면 `proprio_source="none"` 폴백(로봇 base·AG는 raw에서, 카메라는 명목 pose,
  geometric inhand 비활성)으로 라벨을 만든다. 학습 병합(`build_cot_dataset.sh`)에는 parquet이 필요.

## 6. 검증·전체 실행 결과 (2026-09-24)

**검증 (100 task × 2 에피소드, `VALIDATION_2ep.md`)**
* 100/100 task 실행 성공, 디코드 실패 에피소드 0, 총 23분(6병렬).
* 데모 마지막 프레임에서 goal이 GT 라벨로 완전히 만족되는 task **86/100**. 미완성은 `exists` 아래
  conjunction(task 8: 캐비닛 문이 열린 채 종료, 70, 77), transition product의 `ontop`(49 pizza, 62 half egg —
  생성 객체의 scene 이름을 모름), `nextto/under` 임계(5, 6, 89), 입자 `filled/contains`(48).
* task 45는 v1(task 전용 코드) 라벨과 key별 값 일치도 평균 0.985(`reachable/visited countertop`만
  앵커→AABB 거리 변경으로 차이).

**전체 실행 (100 task × 전체 에피소드, `VALIDATION_full.md`, `/data0/hoyong/b1k_data/bg_out`)**
* 20,000/20,000 에피소드(task 38의 1개는 raw/demo 길이 3프레임 차이 → 허용 후 재실행),
  `cot_targets_all.parquet` 7,040,194 행(1 Hz), 출력 9.9 GB, 20병렬 약 4.5시간(CPU 합계 95.7시간).
* belief trace 기준 `final_progress` 평균 0.954, 66 task는 모든 에피소드에서 1.0; task 8만 0.
* `accum_mismatch` 평균 0.131(에피소드 2개일 때와 비슷 — 표본 수가 아니라 규칙 잔차),
  `memory_prefix` 검증 통과율 평균 0.69, `observe_none>0.5`·`subtask_fallback>0.2` task 없음.

## 7. 알려진 한계

* 입자계 predicate은 종료 신호로만 채움(관측 불가).
* exists/forn over conjunction은 카운트 근사; `or` 하위 literal은 성공 시점 강제 대상이 아님.
* task 50–99 category 토큰 → 인스턴스 해소는 휴리스틱.
* 임계값 검증은 task 45(v1 대비 key별 일치도 ≥0.98) 외 개별 확인 없음. `validate.py` 플래그
  (`accum_mismatch>0.15`, `observe_none>0.5`, `final_progress<0.5`)를 task별로 살펴야 한다.
