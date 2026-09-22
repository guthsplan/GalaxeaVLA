"""Offline decoder for the serialized OmniGibson sim state stored in b1k_raw HDF5.

Format (verified against OmniGibson sources — scene_base.serialize, registry_utils.serialize,
entity_prim/rigid_dynamic_prim/xform_prim serialize):

  [scene.pos(3), scene.ori(4), n_registries=2,
   uuid(system_registry), n_systems, <system states...>,
   uuid(object_registry), n_objects,
   per object: uuid, is_asleep, pos(3), quat(4, xyzw), lin_vel(3), ang_vel(3),
               joint_pos(n_j), joint_vel(n_j), <stateful non-kinematic states...>]

uuid = int(float32(md5(name) % 1e8))  (python_utils.get_uuid, deterministic).
Objects are only dumped on frames where they are active (hdf5_data_wrapper dump filter),
so absent objects carry forward their last dumped block (asleep == unchanged).

Known per-object block layouts for task 45 (verified empirically in this dataset):
  hotdog_*            : extras = [Temperature, MaxTemperature]
  fridge_dszchb_0     : extras = [door joint_pos, door joint_vel, Temperature]
  microwave_abzvij_0  : extras = [door joint_pos, door joint_vel, ToggledOn, ?, Temperature]
  robot_r1            : base pose used; the rest (joints/controllers) is read from demo proprio.
"""
from hashlib import md5

import h5py
import numpy as np

AG_MAGIC = np.float32(1e8 + 123456)  # robots/robot.py:_AG_MAGIC
UUID_MIN = 1e6  # empirically uuids are the only values this large besides AG_MAGIC


def get_uuid(name: str) -> int:
    return int(np.float32(int(md5(name.encode()).hexdigest(), 16) % 10**8))


class RawEpisode:
    def __init__(self, path: str, object_names: list[str], robot_name: str = "robot_r1",
                 expected_len: int | None = None):
        self.f = h5py.File(path, "r")
        # a file may hold several attempts (demo_0, demo_1, ...); the published LeRobot
        # episode is the single group whose action count equals the episode length
        # (earlier groups are discarded partial attempts). Verified: demo actions are
        # bit-identical to that group's actions.
        demos = sorted(self.f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        pick = demos[-1]
        if expected_len is not None:
            for dk in demos:
                if self.f[f"data/{dk}"]["action"].shape[0] == expected_len:
                    pick = dk
                    break
        d = self.f[f"data/{pick}"]
        self.demo_group = pick
        self.S = d["state"][:]
        self.SS = d["state_size"][:]
        self.terminated = d["terminated"][:]
        self.n_frames = self.S.shape[0]
        self.names = list(object_names) + [robot_name]
        self.robot_name = robot_name
        self.uuid2name = {get_uuid(n): n for n in self.names}

    def close(self):
        self.f.close()

    def decode(self) -> dict[str, dict[str, np.ndarray]]:
        """Return per-object time series with carry-forward for undumped frames.

        out[name] = dict(pos (N,3), quat (N,4), lin_vel (N,3), extras (N, n_extra), dumped (N,))
        extras are everything after the 14 base floats (variable per object, fixed per name).
        """
        N = self.n_frames
        series: dict[str, dict] = {}
        last: dict[str, np.ndarray] = {}
        extra_len: dict[str, int] = {}
        blocks_per_frame = []
        for fr in range(N):
            s = self.S[fr][: self.SS[fr]].astype(np.float64)
            anchors = np.where((np.abs(s) > UUID_MIN) & (s != AG_MAGIC))[0]
            anchors = anchors[anchors >= 12]  # skip header/registry uuids region conservatively
            frame_blocks = {}
            for j, a in enumerate(anchors):
                u = int(s[a])
                name = self.uuid2name.get(u)
                if name is None:
                    continue
                end = anchors[j + 1] if j + 1 < len(anchors) else len(s)
                frame_blocks[name] = s[a:end]
            blocks_per_frame.append(frame_blocks)
            for name, blk in frame_blocks.items():
                if name not in extra_len:
                    extra_len[name] = max(0, len(blk) - 15)
        for name in self.names:
            ne = extra_len.get(name, 0)
            series[name] = dict(
                pos=np.full((N, 3), np.nan),
                quat=np.full((N, 4), np.nan),
                lin_vel=np.full((N, 3), np.nan),
                extras=np.full((N, ne), np.nan),
                dumped=np.zeros(N, dtype=bool),
            )
        for fr in range(N):
            for name in self.names:
                blk = blocks_per_frame[fr].get(name)
                if blk is not None:
                    last[name] = blk
                    series[name]["dumped"][fr] = True
                blk = last.get(name)
                if blk is None:
                    continue
                sv = series[name]
                sv["pos"][fr] = blk[2:5]
                sv["quat"][fr] = blk[5:9]
                sv["lin_vel"][fr] = blk[9:12]
                ne = sv["extras"].shape[1]
                if ne:
                    ext = blk[15 : 15 + ne]
                    if len(ext) == ne:
                        sv["extras"][fr] = ext
        return series

    def ag_grasps(self) -> list[tuple[int, int, int]]:
        """(frame, arm_idx, target_uuid) for every AG block found. Empty in this dataset."""
        out = []
        for fr in range(self.n_frames):
            s = self.S[fr][: self.SS[fr]]
            for i in np.where(s == AG_MAGIC)[0]:
                out.append((fr, int(s[i + 1]), int(s[i + 2])))
        return out


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
