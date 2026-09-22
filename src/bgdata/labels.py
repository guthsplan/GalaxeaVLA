"""Predicate label extraction for task 45 (cook hot dogs) at 10 Hz.

Sources:
  sim_state — read directly from the decoded raw state vector (joints, ToggledOn, MaxTemperature)
  geom_rule — geometric rule on GT poses (AABB containment, EEF distance, base distance, ...)
  derived   — derived from other labels (visited, inhand OR, inside-countertop=False)
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import inventory
from .config import Thresholds
from .rawstate import RawEpisode, quat_to_rotmat, world_aabb

# object roster for task 45 (annotation object_ids, verified identical across episodes)
FRIDGE = "fridge_dszchb_0"
MICRO = "microwave_abzvij_0"
COUNTER = "countertop_kelker_0"
HOTDOGS = ["hotdog_207", "hotdog_208"]
ROBOT = "robot_r1"
STATE_OBJECTS = [FRIDGE, MICRO] + HOTDOGS  # objects present in the raw state dump

# model bbox extents (m) from bddl3 object_inventory.json['bounding_box_sizes']
EXTENTS = {FRIDGE: np.array([0.6352725, 0.6368088, 1.6838648]),
           MICRO: np.array([0.4223656, 0.4975291, 0.3037398])}

# proprio (observation.state, 61) layout — eval/r1pro.yaml proprio_obs order
SL = dict(base_qvel=(0, 3),
          arm_left_qpos=(3, 10), arm_left_qvel=(10, 17), eef_left_pos=(17, 20),
          eef_left_quat=(20, 24), gripper_left_qpos=(24, 26), gripper_left_qvel=(26, 28),
          arm_right_qpos=(28, 35), arm_right_qvel=(35, 42), eef_right_pos=(42, 45),
          eef_right_quat=(45, 49), gripper_right_qpos=(49, 51), gripper_right_qvel=(51, 53),
          trunk_qpos=(53, 57), trunk_qvel=(57, 61))

CAMS = ["left_realsense_link_camera_0", "right_realsense_link_camera_0", "zed_link_camera_0"]


def load_demo_frames(ep: dict) -> dict[str, np.ndarray]:
    """observation.state + camera extrinsics for one episode, sorted by frame."""
    path = inventory.demo_parquet_path(ep["data_chunk"], ep["data_file"])
    cols = ["episode_index", "frame_index", "observation.state"] + [
        f"observation.robot2cam_pose.{c}" for c in CAMS]
    t = pq.read_table(path, columns=cols,
                      filters=[("episode_index", "=", ep["episode_index"])]).to_pandas()
    t = t.sort_values("frame_index")
    out = dict(state=np.stack(t["observation.state"].values))
    for c in CAMS:
        out[c] = np.stack(t[f"observation.robot2cam_pose.{c}"].values)
    return out


class EpisodeData:
    """Everything needed to compute labels for one episode, at 30 fps."""

    def __init__(self, ep: dict, thr: Thresholds):
        self.ep = ep
        self.thr = thr
        raw = RawEpisode(ep["raw_path"], STATE_OBJECTS, ROBOT, expected_len=ep["length"])
        self.series = raw.decode()
        self.n_raw = raw.n_frames
        self.terminated = raw.terminated
        raw.close()
        self.ann = inventory.load_annotation(ep["annotation_path"])
        self.segments = inventory.segments(self.ann)
        demo = load_demo_frames(ep)
        self.proprio = demo["state"]                       # (N,61)
        self.cams = {c: demo[c] for c in CAMS}             # (N,7) each
        self.N = self.proprio.shape[0]                     # demo frames == raw frames - 1
        assert self.n_raw in (self.N, self.N + 1), \
            f"raw/demo length mismatch: raw {self.n_raw} vs demo {self.N} ({ep['raw_path']})"
        # frame k obs <-> raw state[k]
        self.pos = {n: self.series[n]["pos"][: self.N] for n in self.series}
        self.quat = {n: self.series[n]["quat"][: self.N] for n in self.series}
        self.vel = {n: self.series[n]["lin_vel"][: self.N] for n in self.series}
        self.extras = {n: self.series[n]["extras"][: self.N] for n in self.series}
        # R1 Pro is a holonomic-base robot: the articulation ROOT stays at its spawn pose and
        # base motion lives in 6 virtual joints (x,y,z,rx,ry,rz = first 6 joint_pos entries).
        # robot.get_position_orientation() (the frame of proprio eef poses and robot2cam)
        # returns the base FOOTPRINT: world base = root ∘ virtual joints.
        root_pos, root_quat = self.pos[ROBOT], self.quat[ROBOT]
        vj = self.extras[ROBOT][:, :6]  # x, y, z, rx, ry, rz
        R_root = quat_to_rotmat(root_quat)
        base_pos = root_pos + np.einsum("nij,nj->ni", R_root, vj[:, :3])
        rz = vj[:, 5]
        cz, sz = np.cos(rz), np.sin(rz)
        Rz = np.zeros((self.N, 3, 3))
        Rz[:, 0, 0], Rz[:, 0, 1] = cz, -sz
        Rz[:, 1, 0], Rz[:, 1, 1] = sz, cz
        Rz[:, 2, 2] = 1.0
        R = np.einsum("nij,njk->nik", R_root, Rz)  # rx, ry are ~0 for a ground robot
        self.base_pos = base_pos
        self.eef_world = {}
        for arm in ("left", "right"):
            lo, hi = SL[f"eef_{arm}_pos"]
            rel = self.proprio[:, lo:hi].astype(np.float64)
            self.eef_world[arm] = base_pos + np.einsum("nij,nj->ni", R, rel)
        self.gripper_sum = {
            arm: self.proprio[:, slice(*SL[f"gripper_{arm}_qpos"])].sum(axis=1)
            for arm in ("left", "right")}
        # world camera poses (robot2cam is relative to robot base frame)
        self.cam_pos, self.cam_fwd = {}, {}
        for c in CAMS:
            cp = self.cams[c][:, :3].astype(np.float64)
            cq = self.cams[c][:, 3:7].astype(np.float64)
            self.cam_pos[c] = base_pos + np.einsum("nij,nj->ni", R, cp)
            Rc = np.einsum("nij,njk->nik", R, quat_to_rotmat(cq))
            self.cam_fwd[c] = -Rc[:, :, 2]  # USD camera looks along -Z

    # ---------------- geometric helpers ----------------
    def container_aabb(self, name):
        return world_aabb(self.pos[name], self.quat[name], EXTENTS[name])

    def inside(self, obj, container):
        lo, hi = self.container_aabb(container)
        m = np.array([self.thr.inside_margin_xy] * 2 + [self.thr.inside_margin_z])
        p = self.pos[obj]
        return np.all((p >= lo - m) & (p <= hi + m), axis=1)

    def ontop_box(self, obj, container):
        """obj resting on the TOP face of container's AABB."""
        lo, hi = self.container_aabb(container)
        p = self.pos[obj]
        xy_in = np.all((p[:, :2] >= lo[:, :2]) & (p[:, :2] <= hi[:, :2]), axis=1)
        z_ok = (p[:, 2] > hi[:, 2] - 0.02) & (p[:, 2] < hi[:, 2] + 0.15)
        return xy_in & z_ok & self.at_rest(obj)

    def at_rest(self, obj):
        return np.linalg.norm(self.vel[obj], axis=1) < self.thr.rest_speed

    def door_open(self, name, joint_range):
        jp = self.extras[name][:, 0]
        return jp > self.thr.open_joint_fraction * joint_range

    def object_visible(self, name, pos_override=None):
        """Geometric visibility proxy: inside the view cone of any camera."""
        p = self.pos[name] if pos_override is None else pos_override
        cos_h = np.cos(np.deg2rad(self.thr.vis_half_fov_deg))
        vis = np.zeros(self.N, dtype=bool)
        for c in CAMS:
            d = p - self.cam_pos[c]
            dist = np.linalg.norm(d, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                cosang = np.einsum("ni,ni->n", d, self.cam_fwd[c]) / np.where(dist == 0, 1, dist)
            vis |= (dist < self.thr.vis_max_dist) & (cosang > cos_h)
        return vis


def smooth_bool(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w)
    return (np.convolve(x.astype(float), k, mode="same") / w) > 0.5


def compute_inhand(ed: EpisodeData, thr: Thresholds) -> dict[str, np.ndarray]:
    out = {}
    for h in HOTDOGS:
        for arm in ("left", "right"):
            d = np.linalg.norm(ed.eef_world[arm] - ed.pos[h], axis=1)
            raw = (d < thr.grasp_dist) & (ed.gripper_sum[arm] < thr.gripper_closed_sum)
            out[(h, arm)] = smooth_bool(raw, thr.inhand_smooth_frames * 3)  # window at 30 fps
    return out


def counter_anchor(ed: EpisodeData) -> tuple[np.ndarray, float]:
    """Fitted countertop anchor xy and surface z from 'place on' segment ends."""
    ps = []
    for s in ed.segments:
        if s["skill"] == "place on" and s["manip"] and COUNTER in s["objects"]:
            e = min(s["end"], ed.N - 1)
            ps.append(ed.pos[s["manip"]][e])
    if not ps:
        return None, None
    ps = np.array(ps)
    return np.median(ps[:, :2], axis=0), float(np.median(ps[:, 2]))


def compute_labels(ed: EpisodeData, thr: Thresholds, fitted: dict) -> pd.DataFrame:
    """Full 30 fps boolean series -> 10 Hz rows (task, episode, frame, key, value, visible, tag, source)."""
    N = ed.N
    anchor_xy, z_counter = counter_anchor(ed)
    if z_counter is None:
        z_counter = fitted["counter_z"]
        anchor_xy = np.array(fitted["counter_anchor_xy_default"])

    fridge_range = fitted["fridge_joint_range"]
    micro_range = fitted["micro_joint_range"]

    V = {}   # key -> (value series, visible series, tag, source)
    open_f = ed.door_open(FRIDGE, fridge_range)
    open_m = ed.door_open(MICRO, micro_range)
    vis_fridge = ed.object_visible(FRIDGE)
    vis_micro = ed.object_visible(MICRO)
    V[f"(open {FRIDGE})"] = (open_f, vis_fridge, "appear", "sim_state")
    V[f"(open {MICRO})"] = (open_m, vis_micro, "appear", "sim_state")
    V[f"(toggled_on {MICRO})"] = (ed.extras[MICRO][:, 2] > 0.5, vis_micro, "appear", "sim_state")

    inhand_arm = compute_inhand(ed, thr)
    inside_f = {h: ed.inside(h, FRIDGE) for h in HOTDOGS}
    inside_m = {h: ed.inside(h, MICRO) for h in HOTDOGS}

    for h in HOTDOGS:
        ih_l, ih_r = inhand_arm[(h, "left")], inhand_arm[(h, "right")]
        ih = ih_l | ih_r
        # a hotdog inside a CLOSED container is occluded regardless of the frustum test
        concealed = (inside_f[h] & ~open_f) | (inside_m[h] & ~open_m)
        vis_h = ed.object_visible(h) & ~concealed
        ontop_c = (ed.at_rest(h) & ~inside_f[h] & ~inside_m[h] & ~ih
                   & (np.abs(ed.pos[h][:, 2] - z_counter) < thr.counter_z_band))
        onfloor = (ed.pos[h][:, 2] < thr.floor_z_max) & ed.at_rest(h)
        cooked = ed.extras[h][:, 1] >= thr.cook_temperature_hotdog
        # non-kinematic states of ASLEEP objects are not dumped (their temperature keeps
        # rising in the sim while carry-forward goes stale). BehaviorTask terminates when
        # the goal (forall cooked) holds, so success completes the label from that frame on.
        cooked_src = np.where(cooked, "sim_state", "sim_state").astype(object)
        term = np.where(ed.terminated[: N])[0]
        if len(term):
            override = (~cooked) & (np.arange(N) >= term[0])
            cooked = cooked | override
            cooked_src[override] = "derived"

        V[f"(inside {h} {FRIDGE})"] = (inside_f[h], vis_h & open_f & vis_fridge, "geom", "geom_rule")
        V[f"(inside {h} {MICRO})"] = (inside_m[h], vis_h & open_m & vis_micro, "geom", "geom_rule")
        V[f"(inside {h} {COUNTER})"] = (np.zeros(N, bool), vis_h, "geom", "derived")
        V[f"(ontop {h} {FRIDGE})"] = (ed.ontop_box(h, FRIDGE), vis_h & vis_fridge, "geom", "geom_rule")
        V[f"(ontop {h} {MICRO})"] = (ed.ontop_box(h, MICRO), vis_h & vis_micro, "geom", "geom_rule")
        V[f"(ontop {h} {COUNTER})"] = (ontop_c, vis_h, "geom", "geom_rule")
        V[f"(onfloor {h})"] = (onfloor, vis_h, "geom", "geom_rule")
        V[f"(cooked {h})"] = (cooked, vis_h, "appear", cooked_src)
        V[f"(inhand_left {h})"] = (ih_l, np.ones(N, bool), "robot", "geom_rule")
        V[f"(inhand_right {h})"] = (ih_r, np.ones(N, bool), "robot", "geom_rule")
        V[f"(inhand {h})"] = (ih, np.ones(N, bool), "robot", "derived")

    base_xy = ed.base_pos[:, :2]
    targets = {FRIDGE: ed.pos[FRIDGE][:, :2], MICRO: ed.pos[MICRO][:, :2],
               COUNTER: np.broadcast_to(anchor_xy, (N, 2))}
    for tgt, txy in targets.items():
        reach = np.linalg.norm(base_xy - txy, axis=1) < fitted["reach_dist"]
        visited = np.maximum.accumulate(reach.astype(int)).astype(bool)
        V[f"(reachable {tgt})"] = (reach, np.ones(N, bool), "robot", "geom_rule")
        V[f"(visited {tgt})"] = (visited, np.ones(N, bool), "robot", "derived")

    frames = np.arange(0, N, 3)  # 10 Hz
    rows = []
    for key, (val, vis, tag, source) in V.items():
        src = source[frames] if isinstance(source, np.ndarray) else source
        rows.append(pd.DataFrame(dict(
            task=np.int16(45), episode=np.int64(ed.ep["raw_episode_id"]),
            frame=frames.astype(np.int32), key=key,
            value=val[frames], visible=vis[frames], tag=tag, source=src)))
    return pd.concat(rows, ignore_index=True)


# ---------------- fitting ----------------
def fit_thresholds(eds: list[EpisodeData], thr: Thresholds) -> dict:
    """Fit data-driven thresholds from the given episodes. Returns fitted dict + notes."""
    carry_dists, carry_grips = [], []
    for ed in eds:
        picks = [s for s in ed.segments if s["skill"] == "pick up from" and s["manip"]]
        places = [s for s in ed.segments if s["skill"] in ("place on", "place in") and s["manip"]]
        for p in picks:
            nxt = [q for q in places if q["manip"] == p["manip"] and q["start"] >= p["end"]]
            if not nxt:
                continue
            w0, w1 = p["end"], min(nxt[0]["end"], ed.N - 1)
            h = p["manip"]
            for fr in range(w0, w1, 3):
                d = {arm: np.linalg.norm(ed.eef_world[arm][fr] - ed.pos[h][fr]) for arm in ("left", "right")}
                arm = min(d, key=d.get)
                carry_dists.append(d[arm])
                carry_grips.append(ed.gripper_sum[arm][fr])
    grasp_dist = float(np.percentile(carry_dists, 95))
    grip_closed = float(np.percentile(carry_grips, 95))

    reach_ds, counter_zs, anchors = [], [], []
    fr_range, mi_range = 0.0, 0.0
    for ed in eds:
        fr_range = max(fr_range, float(np.nanmax(ed.extras[FRIDGE][:, 0])))
        mi_range = max(mi_range, float(np.nanmax(ed.extras[MICRO][:, 0])))
        axy, cz = counter_anchor(ed)
        if cz is not None:
            counter_zs.append(cz)
            anchors.append(axy)
        for s in ed.segments:
            if s["skill_type"] == "navigation" or not s["objects"]:
                continue
            tgt = s["objects"][-1]
            fr = min(s["start"], ed.N - 1)
            if tgt in (FRIDGE, MICRO):
                txy = ed.pos[tgt][fr, :2]
            elif tgt == COUNTER and axy is not None:
                txy = axy
            else:
                continue
            reach_ds.append(np.linalg.norm(ed.base_pos[fr, :2] - txy))
    fitted = dict(
        grasp_dist=grasp_dist,
        gripper_closed_sum=grip_closed,
        reach_dist=float(np.percentile(reach_ds, 95)) * thr.reach_margin,
        reach_dist_p95_raw=float(np.percentile(reach_ds, 95)),
        counter_z=float(np.median(counter_zs)),
        counter_anchor_xy_default=[float(x) for x in np.median(np.array(anchors), axis=0)],
        fridge_joint_range=fr_range,
        micro_joint_range=mi_range,
    )
    notes = dict(
        grasp_dist="95th pct of min-arm EEF-hotdog distance over carry windows (pick end -> place end)",
        gripper_closed_sum="95th pct of holding-arm gripper qpos sum over the same windows",
        reach_dist="95th pct of base-target horizontal distance at manipulation segment starts, "
                   f"x{thr.reach_margin} margin (top 5% are systematic arrival distances)",
        counter_z="median resting z of the manipulated hotdog at 'place on countertop' segment ends",
        fridge_joint_range=f"max observed fridge door joint pos = {fr_range:.3f} rad",
        micro_joint_range=f"max observed microwave door joint pos = {mi_range:.3f} rad",
        n_carry_samples=len(carry_dists), n_reach_samples=len(reach_ds),
    )
    thr.grasp_dist = grasp_dist
    thr.gripper_closed_sum = grip_closed
    thr.reach_dist = fitted["reach_dist"]
    thr.counter_z = fitted["counter_z"]
    return fitted, notes
