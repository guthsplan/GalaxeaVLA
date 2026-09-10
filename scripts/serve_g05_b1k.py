#!/usr/bin/env python3
"""BEHAVIOR-1K websocket policy server backed by a G0.5 checkpoint.

Speaks the evaluator protocol used by `omnigibson.eval.eval` (adapted from openpi):
  * on connect the server sends msgpack metadata
  * each request is a flattened obs dict  {"robot_r1::proprio": (61,), "<cam>::rgb": (H,W,3|4), ...}
  * `{"reset": True}` marks an episode boundary (no reply)
  * the reply is {"action": float32[23], "server_timing": {...}}

Inference reuses GalaxeaVLA/scripts/serve_policy.py (setup, build_obs_dict, ChunkedPolicyWrapper) so
pre/post-processing is identical to training. Actions are re-planned every --action-steps env steps
(receding horizon, like the pi0.5 baseline's action_horizon=16).

BEHAVIOR R1Pro layouts (openpi b1k/R1Pro):
  action(23): base_vel[0:3] torso[3:7] left_arm[7:14] left_gripper[14] right_arm[15:22] right_gripper[22]
  state(61):  base_qvel[0:3] left_arm[3:10] left_gripper[24:26] right_arm[28:35] right_gripper[49:51] trunk[53:57]
G0.5 parts: left_arm(7) left_gripper(1) right_arm(7) right_gripper(1) lower_body(7)=[torso(4), base_vel(3)]

Run inside the GalaxeaVLA venv:
  G05_DISABLE_FLASH_ATTN=1 python scripts/serve_g05_b1k.py --ckpt_path <run>/checkpoints/step_30000_model.pt \
      --task-name turning_on_radio --port 8000
"""
import argparse
import asyncio
import collections
import functools
import http
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import msgpack
import numpy as np
import torch
import websockets
import websockets.asyncio.server as _server

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # scripts/ (serve_policy.py)
sys.path.insert(0, str(_HERE.parent))   # repo root

import serve_policy as sp  # noqa: E402
from g05.models.g05.inferencer import PolicyInferencer  # noqa: E402
from g05.utils.checkpoint.ckpt_utils import load_config_from_task_yaml  # noqa: E402

logger = logging.getLogger("serve_g05_b1k")

EMBODIMENT = "behavior_r1pro"
ACTION_DIM = 23
GRIPPER_OPEN_SIGN = float(os.environ.get("B1K_GRIPPER_OPEN_SIGN", "1"))  # action sign that means "open"


# ---------------- msgpack (identical to openpi / omnigibson network_utils) ----------------
def pack_data(obj):
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_data(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_data)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_data)


# ---------------- obs / action conversion ----------------
def _find_key(obs: dict, *needles: str) -> str:
    for k in obs:
        if all(n in k for n in needles):
            return k
    raise KeyError(f"no obs key containing {needles}; keys={sorted(obs)[:20]}")


def _to_chw(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if img.shape[:2] != tuple(hw):
        import cv2
        img = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img.transpose(2, 0, 1))


class G05B1KPolicy:
    def __init__(self, inferencer, processor, task_text: str, action_steps: int, image_hw: dict):
        self.task_text = task_text
        self.image_hw = image_hw
        self.wrapper = sp.ChunkedPolicyWrapper(inferencer, processor, action_steps=action_steps)
        self.n_calls = 0
        self.n_infer = 0
        self.last_gripper = {"left_gripper": None, "right_gripper": None}
        self.missing_counts = collections.Counter()

    def reset(self):
        self.wrapper.reset()
        logger.info(f"episode reset after {self.n_calls} steps / {self.n_infer} model calls; "
                    f"held (absent) parts: {dict(self.missing_counts)}")
        self.n_calls = 0
        self.last_gripper = {"left_gripper": None, "right_gripper": None}
        self.missing_counts = collections.Counter()

    def build_raw_obs(self, obs: dict) -> dict:
        s = np.asarray(obs[_find_key(obs, "::proprio")], dtype=np.float32).reshape(-1)
        assert s.shape[0] == 61, s.shape
        head = obs[_find_key(obs, "zed_link", "::rgb")]
        lw = obs[_find_key(obs, "left_realsense", "::rgb")]
        rw = obs[_find_key(obs, "right_realsense", "::rgb")]
        return {
            "images": {
                "head_rgb": _to_chw(head, self.image_hw["head_rgb"]),
                "left_wrist_rgb": _to_chw(lw, self.image_hw["left_wrist_rgb"]),
                "right_wrist_rgb": _to_chw(rw, self.image_hw["right_wrist_rgb"]),
            },
            "state": {
                "left_arm": s[3:10].copy(),
                "left_gripper": s[24:25].copy(),
                "right_arm": s[28:35].copy(),
                "right_gripper": s[49:50].copy(),
                "lower_body": np.concatenate([s[53:57], s[0:3]]).astype(np.float32),
            },
            "task": self.task_text,
            "frequency": 30,
            "embodiment_type": EMBODIMENT,
        }

    def assemble_action(self, parts: dict, s: np.ndarray) -> np.ndarray:
        """Grouped G0.5 parts -> 23-dim BEHAVIOR action.

        G0.5's ActionCodec omits parts it predicts as no-op (`dropout_noop_parts`), so a missing part
        means "hold": arms/torso hold the current joint state, base velocity 0, grippers keep the last
        commanded value (initially derived from the finger position).
        """
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        if "lower_body" in parts:
            lb = np.asarray(parts["lower_body"], dtype=np.float32).reshape(-1)
            a[0:3] = lb[4:7]            # base velocity
            a[3:7] = lb[0:4]            # torso joints
        else:
            a[0:3] = 0.0
            a[3:7] = s[53:57]
        a[7:14] = np.asarray(parts["left_arm"], dtype=np.float32).reshape(-1) if "left_arm" in parts else s[3:10]
        a[15:22] = np.asarray(parts["right_arm"], dtype=np.float32).reshape(-1) if "right_arm" in parts else s[28:35]
        for name, idx, sidx in (("left_gripper", 14, 24), ("right_gripper", 22, 49)):
            if name in parts:
                v = float(np.asarray(parts[name]).reshape(-1)[0])
                a[idx] = 1.0 if v >= 0 else -1.0
            elif self.last_gripper[name] is not None:
                a[idx] = self.last_gripper[name]
            else:
                a[idx] = GRIPPER_OPEN_SIGN if s[sidx] > 0.025 else -GRIPPER_OPEN_SIGN
            self.last_gripper[name] = a[idx]
        self.missing_counts.update(k for k in ("lower_body", "left_arm", "right_arm", "left_gripper", "right_gripper") if k not in parts)
        return a

    async def act(self, obs: dict) -> np.ndarray:
        s = np.asarray(obs[_find_key(obs, "::proprio")], dtype=np.float32).reshape(-1)
        raw = self.build_raw_obs(obs) if self.wrapper.need_obs else {}
        was_recompute = self.wrapper.need_obs
        parts, _ = await self.wrapper.get_action(raw)
        self.n_calls += 1
        self.n_infer += int(was_recompute)
        return self.assemble_action(parts, s)


# ---------------- websocket server ----------------
def _health_check(connection, request):
    if hasattr(request, "path") and request.path == "/healthz":
        if hasattr(connection, "respond"):
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return http.HTTPStatus.OK, {"Content-Type": "text/plain"}, b"OK\n"
    return None


async def _handler(websocket, policy: G05B1KPolicy, metadata: dict):
    logger.info(f"connection from {websocket.remote_address}")
    packer = Packer()
    await websocket.send(packer.pack(metadata))
    prev_total = None
    while True:
        try:
            t0 = time.monotonic()
            msg = unpackb(await websocket.recv())
            if isinstance(msg, dict) and "reset" in msg:
                policy.reset()
                continue
            t1 = time.monotonic()
            action = await policy.act(msg)
            resp = {"action": action, "server_timing": {"infer_ms": (time.monotonic() - t1) * 1000}}
            if prev_total is not None:
                resp["server_timing"]["prev_total_ms"] = prev_total * 1000
            await websocket.send(packer.pack(resp))
            prev_total = time.monotonic() - t0
            if policy.n_calls % 500 == 1:
                logger.info(f"served {policy.n_calls} steps ({policy.n_infer} model calls)")
        except websockets.ConnectionClosed:
            logger.info("connection closed")
            break
        except Exception:
            logger.error(f"error:\n{traceback.format_exc()}")
            await websocket.close(code=1011, reason="Internal server error")
            raise


async def serve(policy, host, port, metadata):
    handler = functools.partial(_handler, policy=policy, metadata=metadata)
    async with _server.serve(handler, host, port, compression=None, max_size=None,
                             process_request=_health_check) as server:
        logger.info(f"serving on {host}:{port}")
        await server.serve_forever()


def load_task_text(task_name: str, tasks_jsonl: str) -> str:
    with open(tasks_jsonl) as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                if d["task_name"] == task_name:
                    return d["task"]
    raise KeyError(task_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--task-name", required=True)
    ap.add_argument("--tasks-jsonl", default=os.environ.get(
        "B1K_TASKS_JSONL",
        os.path.join(os.environ.get("B1K_SUBSET_DIR", "data/b1k_5task_lerobot"), "meta", "tasks.jsonl")),
        help="LeRobot meta/tasks.jsonl mapping task_name -> instruction "
             "(default: $B1K_TASKS_JSONL or $B1K_SUBSET_DIR/meta/tasks.jsonl)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--action-steps", type=int, default=16, help="env steps per model call (receding horizon)")
    ap.add_argument("--task-yaml", default=str(_HERE.parent / "configs" / "task" / "behavior.yaml"))
    ap.add_argument("--override", nargs="*", help="extra hydra key=value overrides")
    ap.add_argument("--selftest", action="store_true", help="run one inference on a synthetic obs and exit")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    task_text = load_task_text(args.task_name, args.tasks_jsonl)
    logger.info(f"task={args.task_name!r} prompt={task_text!r}")

    # Hydra composition from the task yaml (same path as training); the saved .hydra/config.yaml
    # contains nested interpolations (model.tokenizer -> ${tokenizer}) that OmegaConf alone cannot resolve.
    cfg = load_config_from_task_yaml(args.task_yaml, args.ckpt_path, args.override or [])
    policy_model, processor = sp.setup(cfg, device=args.device)
    inferencer = PolicyInferencer(policy_model, processor, device=args.device)

    p = sp.resolve_processor(processor, {"embodiment_type": EMBODIMENT})
    image_hw = {m["key"]: tuple(m["raw_shape"][1:]) for m in p.shape_meta["images"]}
    logger.info(f"image raw sizes: {image_hw}; action_horizon={getattr(p, 'action_horizon', None)}")
    policy = G05B1KPolicy(inferencer, processor, task_text, args.action_steps, image_hw)

    if args.selftest:
        rng = np.random.default_rng(0)
        state = np.zeros(61, np.float32)
        state[3:10] = [-0.44, 0.05, -0.1, -0.62, 0.38, 0.16, -0.05]
        state[28:35] = [0.44, -0.05, 0.1, -0.62, -0.38, -0.16, 0.05]
        state[53:57] = [0.93, -1.28, -0.5, 0.0]
        obs = {
            "robot_r1::proprio": state,
            "robot_r1::robot_r1:zed_link:Camera:0::rgb": rng.integers(0, 255, (720, 720, 4), dtype=np.uint8),
            "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": rng.integers(0, 255, (480, 480, 4), dtype=np.uint8),
            "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": rng.integers(0, 255, (480, 480, 4), dtype=np.uint8),
        }
        t0 = time.monotonic()
        a0 = asyncio.run(policy.act(obs))
        t_first = time.monotonic() - t0
        t0 = time.monotonic()
        a1 = asyncio.run(policy.act(obs))
        t_cached = time.monotonic() - t0
        np.set_printoptions(precision=3, suppress=True)
        print("first action  :", a0)
        print("second action :", a1)
        print("state arms    :", state[3:10], state[28:35], "torso", state[53:57])
        print(f"infer {t_first*1000:.0f} ms (model call), cached step {t_cached*1000:.1f} ms; "
              f"model calls={policy.n_infer} of {policy.n_calls} steps")
        assert a0.shape == (ACTION_DIM,) and np.isfinite(a0).all()
        print("SELFTEST OK")
        return

    metadata = {"policy": "g05", "ckpt": str(args.ckpt_path), "task": args.task_name,
                "action_steps": args.action_steps}
    asyncio.run(serve(policy, args.host, args.port, metadata))


if __name__ == "__main__":
    main()
