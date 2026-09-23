"""Offline decoder for the serialized OmniGibson sim state stored in the raw HDF5 files.

Frame layout (scene_base.serialize / registry_utils.serialize / entity_prim.serialize):

  [scene.pos(3), scene.ori(4), n_registries(=2),
   uuid(system_registry), n_systems, <system blocks...>,
   uuid(object_registry), n_objects,
   per dumped object: uuid, is_asleep, pos(3), quat(4 xyzw), lin_vel(3), ang_vel(3),
                      joint_pos(n_j), joint_vel(n_j), <non-kinematic states...>]
  robot: same base block, then controller states, then 0..2 assisted-grasp blocks
         [AG_MAGIC, arm_idx, target_uuid, link_idx, parent_pos(3), parent_orn(4),
          child_pos(3), child_orn(4), is_spherical]  (19 floats each)

Only *active* objects are dumped on a frame (hdf5_data_wrapper dump filter), so every
object starts from its scene_file initial state and carries the last dumped block forward.

Decoding is sequential with the per-object sizes known from the scene_file (joint count +
non-kinematic schema), so uuid-valued payloads (AttachedTo, Saturated system ids) cannot be
mistaken for block anchors. Unknown objects (transition products created mid-episode) are
skipped up to the next known anchor.
"""
from __future__ import annotations

import json

import h5py
import numpy as np

from .taskspec import NONKIN_SIZE, TaskSpec, get_uuid

AG_MAGIC = np.float32(1e8 + 123456)  # robots/robot.py:_AG_MAGIC
AG_BLOCK = 19  # MAGIC, arm_idx, uuid, link_idx, parent_pos3, parent_orn4, child_pos3, child_orn4, spherical
UUID_MIN = 1e6
BASE = 15  # uuid + is_asleep + pos3 + quat4 + lin3 + ang3 (dynamic rigid body)
BASE_KIN = 9  # uuid + is_asleep + pos3 + quat4: kinematic-only prims (fixed base, no joints) omit velocities


def base_len(o) -> int:
    return BASE_KIN if (o.fixed_base and o.n_joints == 0) else BASE


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion(s) -> rotation matrix. q: (...,4) -> (...,3,3)."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.where(n == 0, 1, n)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def world_aabb(pos: np.ndarray, quat: np.ndarray, extents: np.ndarray):
    """AABB of an oriented box centered at pos: half = |R| @ (extents/2)."""
    R = quat_to_rotmat(quat)
    half = np.einsum("...ij,j->...i", np.abs(R), np.asarray(extents) / 2.0)
    return pos - half, pos + half


def _nonkin_len(names: list[str], s: np.ndarray, start: int) -> int:
    """Total float count of the non-kinematic states starting at s[start]."""
    i = start
    for nm in names:
        k = NONKIN_SIZE.get(nm)
        if k is None:  # variable: n_systems + (uuid, value) pairs (+2 default limits for Saturated)
            n = int(s[i]) if i < len(s) else 0
            head = 3 if nm == "Saturated" else 1
            k = head + 2 * max(0, n)
        i += k
    return i - start


class RawEpisode:
    def __init__(self, path: str, spec: TaskSpec, expected_len: int | None = None,
                 extra_names: list[str] | None = None):
        self.f = h5py.File(path, "r")
        self.spec = spec
        # a file may hold several attempts (demo_0, demo_1, ...); the published LeRobot episode
        # is the group whose action count equals the episode length.
        demos = sorted(self.f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        # groups without `action` are aborted attempts (state only) -> never the published one
        complete = [dk for dk in demos if "action" in self.f[f"data/{dk}"]] or demos
        pick = complete[-1]
        if expected_len is not None:
            for dk in complete:
                if self.f[f"data/{dk}"]["action"].shape[0] == expected_len:
                    pick = dk
                    break
        d = self.f[f"data/{pick}"]
        self.demo_group = pick
        self.S = d["state"]
        self.SS = d["state_size"][:]
        self.terminated = d["terminated"][:] if "terminated" in d else np.zeros(self.S.shape[0], bool)
        self.n_frames = self.S.shape[0]
        self.uuid2name = {o.uuid: n for n, o in spec.objects.items()}
        for n in extra_names or []:  # transition products named by the annotator
            self.uuid2name.setdefault(get_uuid(n), n)
        self.registry_uuids = {get_uuid("object_registry"), get_uuid("system_registry")}
        self.stats = dict(frames=self.n_frames, unknown_blocks=0, resync=0, ag_frames=0)

    def close(self):
        self.f.close()

    # -----------------------------------------------------------------------------------
    def _is_anchor(self, v: float) -> bool:
        # uuids are integers in [0, 1e8) -- some are small (170603), so test set membership
        # of the exact integer value rather than magnitude
        return v == int(v) and np.float32(v) != AG_MAGIC and int(v) in self.uuid2name

    def _next_anchor(self, s: np.ndarray, i: int) -> int:
        j = i
        while j < len(s):
            if self._is_anchor(s[j]):
                return j
            j += 1
        return len(s)

    def parse_frame(self, s: np.ndarray) -> tuple[dict[str, np.ndarray], list[tuple[int, int]]]:
        """-> ({name: block}, [(arm_idx, target_uuid), ...])"""
        blocks: dict[str, np.ndarray] = {}
        ag: list[tuple[int, int]] = []
        # header: scene pose (7), n_registries (1), then registries in order
        i = 8
        # system registry: uuid, n_systems, system blocks (sizes unknown) -> skip to object registry
        if i < len(s) and int(s[i]) == get_uuid("system_registry"):
            n_sys = int(s[i + 1])
            i += 2
            if n_sys > 0:
                j = i
                while j < len(s) and int(s[j]) != get_uuid("object_registry"):
                    j += 1
                i = j
        if i < len(s) and int(s[i]) == get_uuid("object_registry"):
            i += 2  # uuid, n_objects
        robot = self.spec.robot
        while i < len(s):
            v = s[i]
            if np.float32(v) == AG_MAGIC:  # stray AG block (robot handled below); skip
                i += AG_BLOCK
                continue
            name = self.uuid2name.get(int(v)) if v == int(v) else None
            if name is None:
                self.stats["unknown_blocks"] += 1
                i = self._next_anchor(s, i + 1)
                continue
            o = self.spec.objects.get(name)
            if o is None:  # annotation-named transition product: unknown layout
                j = self._next_anchor(s, i + 1)
                blocks[name] = s[i:j]
                i = j
                continue
            if name == robot:
                # base + joints + controllers: unknown controller size -> find the end by
                # the next anchor, then peel trailing AG blocks (they sit inside the robot block)
                j = self._next_anchor(s, i + 1)
                # AG blocks may contain the target uuid (an anchor!) -> pull j back to MAGIC
                k = i + BASE + 2 * o.n_joints
                m = k
                while m < j:
                    if np.float32(s[m]) == AG_MAGIC:
                        break
                    m += 1
                if m < j:  # AG block(s) start at m
                    blk_end = m
                    p = m
                    while p + AG_BLOCK <= len(s) and np.float32(s[p]) == AG_MAGIC:
                        ag.append((int(s[p + 1]), int(s[p + 2])))
                        p += AG_BLOCK
                    blocks[name] = s[i:blk_end]
                    i = p
                    if ag:
                        self.stats["ag_frames"] += 1
                else:
                    blocks[name] = s[i:j]
                    i = j
                continue
            end = i + base_len(o) + 2 * o.n_joints
            end += _nonkin_len(o.non_kin, s, end)
            if end > len(s) or (end < len(s) and not (self._is_anchor(s[end]) or np.float32(s[end]) == AG_MAGIC)):
                # layout mismatch: fall back to the next anchor
                end2 = self._next_anchor(s, i + 1)
                self.stats["resync"] += 1
                end = end2
            blocks[name] = s[i:end]
            i = end
        return blocks, ag

    # -----------------------------------------------------------------------------------
    def decode(self, names: list[str] | None = None, stride: int = 1) -> dict:
        """Per-object time series with carry-forward from the scene_file initial state.

        out[name] = dict(pos (N,3), quat (N,4), lin_vel (N,3), joint_pos (N,nj),
                         non_kin {state: (N,k)}, dumped (N,))
        out["__ag__"] = list over frames of {arm_idx: target_name}
        """
        spec = self.spec
        names = list(names) if names is not None else list(spec.objects)
        N = self.n_frames
        series: dict[str, dict] = {}
        for n in names:
            o = spec.objects.get(n)
            if o is None:
                series[n] = dict(pos=np.full((N, 3), np.nan), quat=np.full((N, 4), np.nan),
                                 lin_vel=np.zeros((N, 3)), joint_pos=np.zeros((N, 0)),
                                 non_kin={}, dumped=np.zeros(N, bool))
                continue
            nk = {}
            for st in o.non_kin:
                k = NONKIN_SIZE.get(st)
                nk[st] = np.full((N, k if k else 1), np.nan)
            series[n] = dict(pos=np.tile(o.init_pos, (N, 1)), quat=np.tile(o.init_quat, (N, 1)),
                             lin_vel=np.zeros((N, 3)), joint_pos=np.tile(o.init_joint_pos, (N, 1)),
                             non_kin=nk, dumped=np.zeros(N, bool))
            # initial non-kin from scene_file
            for st, arr in nk.items():
                v0 = self._init_nonkin_value(o.init_non_kin.get(st), st)
                if v0 is not None and len(v0) == arr.shape[1]:
                    arr[:] = v0
        ag_series: list[dict[int, str]] = [dict() for _ in range(N)]
        last: dict[str, np.ndarray] = {}
        S = self.S
        want = set(names)
        for fr in range(N):
            s = np.asarray(S[fr][: self.SS[fr]], dtype=np.float64)
            blocks, ag = self.parse_frame(s)
            for arm, u in ag:
                nm = self.uuid2name.get(u)
                if nm:
                    ag_series[fr][arm] = nm
            for nm, blk in blocks.items():
                if nm in want:
                    last[nm] = blk
                    series[nm]["dumped"][fr] = True
            for nm, blk in last.items():
                sv = series[nm]
                o = spec.objects.get(nm)
                b = base_len(o) if o is not None else BASE
                if len(blk) < min(b, 9):
                    continue
                sv["pos"][fr] = blk[2:5]
                sv["quat"][fr] = blk[5:9]
                if b == BASE and len(blk) >= 12:
                    sv["lin_vel"][fr] = blk[9:12]
                if o is None:
                    continue
                nj = o.n_joints
                if nj and len(blk) >= b + nj:
                    sv["joint_pos"][fr] = blk[b:b + nj]
                k = b + 2 * nj
                for st in o.non_kin:
                    size = NONKIN_SIZE.get(st)
                    if size is None:
                        n = int(blk[k]) if k < len(blk) else 0
                        sv["non_kin"][st][fr] = float(n)
                        k += (3 if st == "Saturated" else 1) + 2 * max(0, n)
                        continue
                    if k + size <= len(blk):
                        sv["non_kin"][st][fr] = blk[k:k + size]
                    k += size
        series["__ag__"] = ag_series
        return series

    @staticmethod
    def _init_nonkin_value(d, st: str):
        if d is None:
            return None
        if st == "Temperature":
            return [float(d.get("temperature", np.nan))]
        if st == "MaxTemperature":
            return [float(d.get("max_temperature", np.nan))]
        if st == "ToggledOn":
            return [float(bool(d.get("value", False))), float(d.get("hand_in_marker_steps", 0))]
        if st == "AttachedTo":
            return [float(d.get("attached_obj_uuid", -1))]
        if st == "SlicerActive":
            return [float(bool(d.get("value", True))), float(bool(d.get("previously_touching", False))),
                    float(d.get("delay_counter", 0))]
        if st in ("ParticleRemover", "ParticleApplier"):
            return [float(d.get("current_step", 0))]
        if st in ("Saturated", "ModifiedParticles"):
            return [float(d.get("n_systems", 0))]
        return None


def episode_config(path: str) -> tuple[dict, dict]:
    with h5py.File(path, "r") as h:
        return json.loads(h["data"].attrs["scene_file"]), json.loads(h["data"].attrs["config"])
