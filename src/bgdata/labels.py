"""Predicate label extraction at 10 Hz for any of the 100 tasks.

Sources:
  sim_state — read from the decoded raw state (joints -> open, ToggledOn, Temperature/
              MaxTemperature -> cooked/frozen/on_fire, AttachedTo, assisted-grasp -> inhand)
  geom_rule — geometric rule on GT poses + scaled bbox extents (inside/ontop/nextto/under/
              touching/onfloor, EEF distance for inhand without AG, base distance for reachable)
  derived   — derived from other labels (visited, inhand OR) or from the termination signal
              (particle predicates: covered/filled/contains/real, which the raw state does not
              carry; they take the BDDL :init value until the episode terminates successfully)

The set of predicate keys is data-driven: every grounded atom of the BDDL goal/init, plus the
relations implied by the skill annotations (pick up X from Y -> in_or_on X Y, ...), plus the
unary abilities the synset actually has (cookable -> cooked, openable -> open, ...).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import inventory, operators
from .config import Thresholds
from .goal import TaskGoal, key, parse_key
from .rawstate import RawEpisode, quat_to_rotmat, world_aabb
from .taskspec import TaskSpec, flatten_objects

# proprio (observation.state, 61) layout — eval/r1pro.yaml proprio_obs order
SL = dict(base_qvel=(0, 3),
          arm_left_qpos=(3, 10), arm_left_qvel=(10, 17), eef_left_pos=(17, 20),
          eef_left_quat=(20, 24), gripper_left_qpos=(24, 26), gripper_left_qvel=(26, 28),
          arm_right_qpos=(28, 35), arm_right_qvel=(35, 42), eef_right_pos=(42, 45),
          eef_right_quat=(45, 49), gripper_right_qpos=(49, 51), gripper_right_qvel=(51, 53),
          trunk_qpos=(53, 57), trunk_qvel=(57, 61))
CAMS = ["left_realsense_link_camera_0", "right_realsense_link_camera_0", "zed_link_camera_0"]
ARMS = ("left", "right")
PARTICLE_PREDS = {"covered", "filled", "contains", "real", "saturated", "insource"}
# skill -> relations (pred, obj_arg_idx, target_arg_idx) it implies between its arguments
SKILL_RELATIONS = {
    "pick up from": [("inside", 0, 1), ("ontop", 0, 1)],
    "place in": [("inside", 0, 1)], "place in next to": [("inside", 0, 1), ("nextto", 0, 2)],
    "place on": [("ontop", 0, 1)], "place on next to": [("ontop", 0, 1), ("nextto", 0, 2)],
    "place under": [("under", 0, 1)], "push to": [("ontop", 0, 1), ("nextto", 0, 1)],
    "insert": [("inside", 0, 1)], "attach": [("attached", 0, 1)], "hang": [("attached", 0, 1)],
    "sweep off": [("ontop", 0, 1)],
}
DOOR_SKILLS = {"open door", "close door", "open drawer", "close drawer", "open lid", "close lid",
               "pull tray", "push tray"}
SWITCH_SKILLS = {"turn on switch", "turn off switch", "press"}


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


def smooth_bool(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w)
    return (np.convolve(x.astype(float), k, mode="same") / w) > 0.5


class EpisodeData:
    """Everything needed to compute labels for one episode, at 30 fps."""

    def __init__(self, ep: dict, spec: TaskSpec, thr: Thresholds):
        self.ep = ep
        self.spec = spec
        self.thr = thr
        self.ann = inventory.load_annotation(ep["annotation_path"])
        self.raw_segments = inventory.segments(self.ann)
        ann_tokens = {t for s in self.raw_segments for t in flatten_objects(s["objects"])}
        extra = [t for t in ann_tokens if t not in spec.objects]
        raw = RawEpisode(ep["raw_path"], spec, expected_len=ep["length"], extra_names=extra)
        names = sorted(set(spec.relevant) | {spec.robot} | set(extra))
        self.series = raw.decode(names=names)
        self.decode_stats = dict(raw.stats)
        self.n_raw = raw.n_frames
        self.terminated = raw.terminated
        raw.close()
        try:
            demo = load_demo_frames(ep)
            self.proprio = demo["state"]                   # (N,61)
            self.cams = {c: demo[c] for c in CAMS}
            self.N = self.proprio.shape[0]                 # demo frames == raw frames - 1
            self.proprio_source = "demo"
        except FileNotFoundError:
            # LeRobot parquet not downloaded for this task: the raw state still gives the robot
            # base and the assisted-grasp constraints; EEF/camera poses are approximated.
            self.N = max(1, self.n_raw - 1) if ep.get("length") is None else min(int(ep["length"]), self.n_raw)
            self.proprio = None
            self.cams = None
            self.proprio_source = "none"
        # demo frames == raw frames - 1 normally; a handful of recordings differ by a few frames
        assert 0 <= self.n_raw - self.N <= 5, \
            f"raw/demo length mismatch: raw {self.n_raw} vs demo {self.N} ({ep['raw_path']})"
        N = self.N
        self.pos = {n: self.series[n]["pos"][:N] for n in names}
        self.quat = {n: self.series[n]["quat"][:N] for n in names}
        self.vel = {n: np.where(self.series[n]["dumped"][:N, None], self.series[n]["lin_vel"][:N], 0.0)
                    for n in names}  # not dumped == asleep == at rest (carry-forward keeps a stale velocity)
        self.joints = {n: self.series[n]["joint_pos"][:N] for n in names}
        self.nonkin = {n: {k: v[:N] for k, v in self.series[n]["non_kin"].items()} for n in names}
        self.ag = self.series["__ag__"][:N]
        self.has_ag = any(self.ag)
        # R1 Pro is a holonomic-base robot: the articulation ROOT stays at its spawn pose and
        # base motion lives in 6 virtual joints (x,y,z,rx,ry,rz = first 6 joint_pos entries).
        rob = spec.robot
        root_pos, root_quat = self.pos[rob], self.quat[rob]
        vj = self.joints[rob][:, :6]
        R_root = quat_to_rotmat(root_quat)
        base_pos = root_pos + np.einsum("nij,nj->ni", R_root, vj[:, :3])
        rz = vj[:, 5]
        cz, sz = np.cos(rz), np.sin(rz)
        Rz = np.zeros((N, 3, 3))
        Rz[:, 0, 0], Rz[:, 0, 1] = cz, -sz
        Rz[:, 1, 0], Rz[:, 1, 1] = sz, cz
        Rz[:, 2, 2] = 1.0
        R = np.einsum("nij,njk->nik", R_root, Rz)
        self.base_pos = base_pos
        self.eef_world = {}
        self.cam_pos, self.cam_fwd = {}, {}
        if self.proprio is not None:
            for arm in ARMS:
                lo, hi = SL[f"eef_{arm}_pos"]
                rel = self.proprio[:, lo:hi].astype(np.float64)
                self.eef_world[arm] = base_pos + np.einsum("nij,nj->ni", R, rel)
            self.gripper_sum = {arm: self.proprio[:, slice(*SL[f"gripper_{arm}_qpos"])].sum(axis=1)
                                for arm in ARMS}
            for c in CAMS:
                cp = self.cams[c][:, :3].astype(np.float64)
                cq = self.cams[c][:, 3:7].astype(np.float64)
                self.cam_pos[c] = base_pos + np.einsum("nij,nj->ni", R, cp)
                Rc = np.einsum("nij,njk->nik", R, quat_to_rotmat(cq))
                self.cam_fwd[c] = -Rc[:, :, 2]  # USD camera looks along -Z
        else:
            # nominal R1 Pro head camera: ~1.5 m above the base footprint, looking along base +x,
            # tilted down; wrist cameras unknown -> a slightly lower forward cone
            fwd = np.einsum("nij,j->ni", R, np.array([1.0, 0.0, -0.45]))
            fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
            for c, dz in zip(CAMS, (1.0, 1.0, 1.5)):
                self.cam_pos[c] = base_pos + np.array([0.0, 0.0, dz])
                self.cam_fwd[c] = fwd
            for arm in ARMS:  # EEF unknown: park it above the base so geometric inhand never fires
                self.eef_world[arm] = base_pos + np.array([0.0, 0.0, 1.0])
            self.gripper_sum = {arm: np.full(N, 1e9) for arm in ARMS}
        # grasped objects are attached to the gripper; their dumped pose may lag -> use the EEF
        if self.has_ag:
            for fr, arms in enumerate(self.ag):
                for arm_idx, name in arms.items():
                    if name in self.pos and spec.objects.get(name) and spec.objects[name].movable:
                        self.pos[name][fr] = self.eef_world[ARMS[arm_idx]][fr]
        self.segments = self._resolve_segments()

    # ---------------- annotation token -> instance --------------------------------
    def _displacement(self, name, a, b):
        p = self.pos.get(name)
        if p is None:
            return 0.0
        a, b = max(0, min(a, self.N - 1)), max(0, min(b, self.N - 1))
        return float(np.nansum(np.linalg.norm(np.diff(p[a:b + 1], axis=0), axis=1)))

    def _resolve_segments(self) -> list[dict]:
        """Category tokens (tasks 50-99) -> the instance that actually took part in the segment:
        manipulated object = largest displacement in the window; targets = nearest to the
        manipulated object (or to the robot base) at the segment end."""
        spec = self.spec
        out = []
        for s in self.raw_segments:
            toks = flatten_objects(s["objects"])
            manip_tok = s["manip"] if isinstance(s["manip"], str) else (
                flatten_objects([s["manip"]])[0] if s["manip"] else None)
            a, b = s["start"], min(s["end"], self.N - 1)
            resolved, manip_name = [], None
            for j, tok in enumerate(toks):
                cands = spec.token_candidates.get(tok) or ([tok] if tok in spec.objects else [])
                cands = [c for c in cands if c != spec.robot]
                if not cands:
                    resolved.append(tok if tok in self.pos else None)
                    continue
                if len(cands) == 1:
                    pick = cands[0]
                elif j == 0 or tok == manip_tok:
                    pick = max(cands, key=lambda c: self._displacement(c, a, b))
                else:
                    ref = self.pos[manip_name][b] if manip_name and manip_name in self.pos else self.base_pos[b]
                    pick = min(cands, key=lambda c: float(np.nan_to_num(
                        np.linalg.norm(self.pos[c][b] - ref), nan=1e9)))
                resolved.append(pick)
                if tok == manip_tok and manip_name is None:
                    manip_name = pick
            if manip_name is None and manip_tok is not None:
                manip_name = resolved[0] if resolved else None
            out.append(dict(s, objects=[r for r in resolved if r is not None],
                            raw_objects=toks, manip=manip_name if manip_tok else None,
                            resolved_all=all(r is not None for r in resolved)))
        return out

    # ---------------- geometric helpers ----------------
    def aabb(self, name):
        o = self.spec.objects.get(name)
        if o is None or o.extents is None:
            p = self.pos[name]
            return p - 0.05, p + 0.05
        lo, hi = world_aabb(self.pos[name], self.quat[name], o.extents)
        ext = self.drawer_extension(name)
        if ext is not None:  # an open drawer sticks out of the closed-body AABB
            lo = lo.copy(); hi = hi.copy()
            lo[:, :2] -= ext[:, None]
            hi[:, :2] += ext[:, None]
        return lo, hi

    def drawer_extension(self, name):
        """(N,) horizontal extension of a container's AABB from its prismatic joints (drawers).
        Joint types are not in the metadata: a joint whose observed travel stays below
        `drawer_max_travel` (m) is taken as prismatic (doors swing >= ~1 rad in the demos)."""
        if not hasattr(self, "_drawer_ext"):
            self._drawer_ext = {}
        if name in self._drawer_ext:
            return self._drawer_ext[name]
        q = self.joints.get(name)
        ext = None
        if q is not None and q.shape[1] and self.spec.objects[name].has("openable"):
            with np.errstate(invalid="ignore"):
                span = np.nanmax(q, axis=0) - np.nanmin(q, axis=0)
                first = np.argmax(np.isfinite(q[:, 0])) if np.isfinite(q[:, 0]).any() else 0
                closed = q[first]  # drawers start closed in every task instance
            pris = np.isfinite(span) & (span > 0.02) & (span < self.thr.drawer_max_travel)
            if pris.any():
                ext = np.nanmax(np.abs(q[:, pris] - closed[pris]), axis=1)
                ext = np.nan_to_num(ext, nan=0.0)
        self._drawer_ext[name] = ext
        return ext

    def is_container(self, name) -> bool:
        """fillable/openable by ability, or used as the container of an `inside` literal in the
        task's BDDL (open boxes such as storage_box/toy_box carry neither ability)."""
        o = self.spec.objects.get(name)
        if o is None:
            return False
        if o.has("openable") or name in self.bddl_containers:
            return True
        # fillable but flat (plate, cutting board, tray rim < 6 cm): rigid bodies rest ON it
        return o.has("fillable") and o.extents is not None and float(o.extents[2]) >= self.thr.container_min_height

    @property
    def bddl_containers(self) -> set[str]:
        if not hasattr(self, "_bddl_containers"):
            from .goal import TaskGoal
            tg = TaskGoal(self.spec.bddl_objects, self.spec.bddl_init, self.spec.bddl_goal,
                          self.spec.inst_to_name)
            self._bddl_containers = {args[1] for pred, args in tg.atoms()
                                     if pred in ("inside", "filled", "contains") and len(args) == 2}
        return self._bddl_containers

    def inside(self, obj, container):
        if not self.is_container(container):
            return np.zeros(self.N, bool)  # a countertop's AABB is solid: nothing is "inside" it
        lo, hi = self.aabb(container)
        m = np.array([self.thr.inside_margin_xy] * 2 + [self.thr.inside_margin_z])
        p = self.pos[obj]
        # the center must lie below the container's top face (a cupcake whose center pokes above
        # a low tray/plate is ontop, not inside); margin applies to the sides and the bottom only
        return (np.all(p >= lo - m, axis=1) & np.all(p[:, :2] <= hi[:, :2] + m[:2], axis=1)
                & (p[:, 2] <= hi[:, 2]))

    def ontop(self, obj, support):
        lo, hi = self.aabb(support)
        olo, ohi = self.aabb(obj)
        c = self.pos[obj]
        xy_in = np.all((c[:, :2] >= lo[:, :2] - 0.02) & (c[:, :2] <= hi[:, :2] + 0.02), axis=1)
        top_ok = (olo[:, 2] > hi[:, 2] - self.thr.ontop_z_tol) & (olo[:, 2] < hi[:, 2] + self.thr.ontop_z_gap)
        if self.is_container(support):  # resting on a shelf INSIDE the container is `inside`
            return xy_in & top_ok & self.at_rest(obj) & ~self.inside(obj, support)
        # a solid support (desk with a keyboard tray, stove, sink rim): any resting level inside
        # its vertical span counts, since nothing can be `inside` it
        z_ok = top_ok | ((olo[:, 2] > lo[:, 2] + self.thr.ontop_z_tol) & (olo[:, 2] <= hi[:, 2]))
        return xy_in & z_ok & self.at_rest(obj)

    def xy_dist_to_aabb(self, point_xy: np.ndarray, name: str) -> np.ndarray:
        """Horizontal distance from points (N,2) to the object's xy-AABB (0 when inside)."""
        lo, hi = self.aabb(name)
        d = np.maximum(0.0, np.maximum(lo[:, :2] - point_xy, point_xy - hi[:, :2]))
        return np.linalg.norm(np.nan_to_num(d, nan=1e9), axis=1)

    def under(self, obj, other):
        lo, hi = self.aabb(other)
        c = self.pos[obj]
        xy_in = np.all((c[:, :2] >= lo[:, :2]) & (c[:, :2] <= hi[:, :2]), axis=1)
        return xy_in & (c[:, 2] < lo[:, 2] + 0.02)

    def aabb_gap(self, a, b):
        alo, ahi = self.aabb(a)
        blo, bhi = self.aabb(b)
        gap = np.maximum(0.0, np.maximum(alo, blo) - np.minimum(ahi, bhi))
        return np.linalg.norm(gap, axis=1), (ahi - alo), (bhi - blo)

    def nextto(self, a, b):
        """OmniGibson NextTo: AABB gap < mean(dims_a + dims_b) / 6 and horizontal adjacency."""
        dist, da, db = self.aabb_gap(a, b)
        avg = np.mean(da + db, axis=1)
        alo, ahi = self.aabb(a)
        blo, bhi = self.aabb(b)
        vert_overlap = (np.minimum(ahi[:, 2], bhi[:, 2]) - np.maximum(alo[:, 2], blo[:, 2])) > 0
        return (dist < avg / 6.0) & vert_overlap & ~self.inside(a, b)

    def touching(self, a, b):
        dist, _, _ = self.aabb_gap(a, b)
        return dist < self.thr.touch_gap

    def attached(self, a, b):
        ta = self.nonkin.get(a, {}).get("AttachedTo")
        tb = self.nonkin.get(b, {}).get("AttachedTo")
        ua, ub = self.spec.objects[a].uuid, self.spec.objects[b].uuid
        v = np.zeros(self.N, bool)
        if ta is not None:
            v |= ta[:, 0] == ub
        if tb is not None:
            v |= tb[:, 0] == ua
        return v

    def at_rest(self, obj):
        return np.linalg.norm(np.nan_to_num(self.vel[obj]), axis=1) < self.thr.rest_speed

    def joint_open(self, name, ranges: dict):
        o = self.spec.objects[name]
        q = self.joints[name]
        if q.shape[1] == 0:
            return np.zeros(self.N, bool)
        idx = [i for i in o.openable_joints if i < q.shape[1]] or list(range(q.shape[1]))
        r = ranges.get(name)
        if r is None:  # per-episode fallback
            r = dict(lo=np.nanmin(q, axis=0), hi=np.nanmax(q, axis=0))
        lo, hi = np.asarray(r["lo"]), np.asarray(r["hi"])
        span = hi - lo
        open_any = np.zeros(self.N, bool)
        for i in idx:
            if span[i] < 1e-4:
                continue
            # closed end = the extreme the joint spends most time at in the fit episodes
            closed = lo[i] if r.get("closed_low", [True] * len(lo))[i] else hi[i]
            frac = np.abs(q[:, i] - closed) / span[i]
            open_any |= frac > self.thr.open_joint_fraction
        return open_any

    def object_visible(self, name):
        """Geometric visibility proxy: the object's closest AABB point lies in a camera's view cone
        (for a long countertop the center can be metres away while its edge is in front of us)."""
        p = self.pos[name]
        lo, hi = self.aabb(name)
        cos_h = np.cos(np.deg2rad(self.thr.vis_half_fov_deg))
        vis = np.zeros(self.N, dtype=bool)
        for c in CAMS:
            # candidate points: the object's center and the box point nearest to a point 1 m ahead of
            # the camera (the point nearest the camera itself is straight below it for a fixture the
            # robot stands next to and would fall outside the cone)
            ahead = self.cam_pos[c] + self.cam_fwd[c]
            for q in (p, np.clip(ahead, lo, hi)):
                d = q - self.cam_pos[c]
                dist = np.linalg.norm(d, axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cosang = np.einsum("ni,ni->n", d, self.cam_fwd[c]) / np.where(dist == 0, 1, dist)
                vis |= (dist < self.thr.vis_max_dist) & (cosang > cos_h)
        return vis & ~np.isnan(p[:, 0])


# ---------------- key universe ----------------
def key_universe(spec: TaskSpec, tg: TaskGoal, segs_by_ep: dict[int, list[dict]]) -> dict:
    """{'unary': {(pred, obj)}, 'binary': {(pred, a, b)}, 'particle': {(pred, obj, substance)},
        'targets': {names used as move/manipulation targets}}"""
    objs = spec.objects

    def cands(tok):
        if tok in objs and tok != spec.robot:
            return [tok]
        return [c for c in spec.token_candidates.get(tok, []) if c != spec.robot]
    unary, binary, particle, targets = set(), set(), set(), set()
    for pred, args in tg.atoms():
        if pred in PARTICLE_PREDS:
            # `real` of a transition product (diced__onion.n.01_1) names a future object that has no
            # scene instance yet: keep it as a pseudo-object so the goal line can complete
            if args and (args[0] in objs or (pred == "real" and len(args) == 1)):
                particle.add((pred, args[0], args[1] if len(args) > 1 else ""))
        elif len(args) == 1 and args[0] in objs:
            unary.add((pred, args[0]))
        elif len(args) == 2 and args[0] in objs and args[1] in objs:
            binary.add((pred, args[0], args[1]))
    for segs in segs_by_ep.values():
        for s in segs:
            toks = flatten_objects(s.get("raw_objects", s["objects"]))
            ob = [cands(t) for t in toks]
            for pred, i, j in SKILL_RELATIONS.get(s["skill"], []):
                if i < len(ob) and j < len(ob):
                    for a in ob[i]:
                        for b in ob[j]:
                            if a != b:
                                binary.add((pred, a, b))
            if s["skill"] in DOOR_SKILLS and ob:
                for a in ob[0]:
                    unary.add(("open", a))
            if s["skill"] in SWITCH_SKILLS and ob:
                for a in ob[0]:
                    unary.add(("toggled_on", a))
            if ob:
                last = ob[-1] if len(ob) > 1 else ob[0]
                for a in last:
                    if not objs[a].movable or s["skill"] == "move to":
                        targets.add(a)
    # abilities the object actually has
    for n in spec.relevant:
        o = objs[n]
        if o.has("cookable"):
            unary.add(("cooked", n))
        if o.has("openable") and o.n_joints > 0:
            unary.add(("open", n))
        if "ToggledOn" in o.non_kin:
            unary.add(("toggled_on", n))
    unary = {(p, n) for p, n in unary if p in ("open", "toggled_on", "cooked", "frozen", "on_fire", "hot", "onfloor")}
    binary = {(p, a, b) for p, a, b in binary if p in ("inside", "ontop", "nextto", "under", "touching", "attached")}
    return dict(unary=unary, binary=binary, particle=particle, targets=targets)


# ---------------- inhand ----------------
def compute_inhand(ed: EpisodeData, thr: Thresholds, movables: list[str]) -> dict:
    out = {}
    N = ed.N
    if ed.has_ag:
        for h in movables:
            for ai, arm in enumerate(ARMS):
                v = np.array([arms.get(ai) == h for arms in ed.ag], dtype=bool)
                out[(h, arm)] = v
        return out
    for h in movables:
        for arm in ARMS:
            d = np.linalg.norm(ed.eef_world[arm] - ed.pos[h], axis=1)
            raw = (d < thr.grasp_dist) & (ed.gripper_sum[arm] < thr.gripper_closed_sum)
            out[(h, arm)] = smooth_bool(np.nan_to_num(raw), thr.inhand_smooth_frames * 3)
    return out


# ---------------- labels ----------------
def compute_labels(ed: EpisodeData, thr: Thresholds, fitted: dict, universe: dict,
                   tg: TaskGoal) -> pd.DataFrame:
    spec, N = ed.spec, ed.N
    objs = spec.objects
    V = {}   # key -> (value series, visible series, tag, source)
    vis_cache = {}

    def vis(n):
        if n not in vis_cache:
            vis_cache[n] = ed.object_visible(n)
        return vis_cache[n]

    ranges = fitted.get("joint_ranges", {})
    open_of = {}
    for pred, n in sorted(universe["unary"]):
        o = objs[n]
        if pred == "open":
            v = ed.joint_open(n, ranges)
            open_of[n] = v
            V[key("open", [n])] = (v, vis(n), "appear", "sim_state")
        elif pred == "toggled_on":
            t = ed.nonkin[n].get("ToggledOn")
            if t is None:
                continue
            V[key("toggled_on", [n])] = (np.nan_to_num(t[:, 0]) > 0.5, vis(n), "appear", "sim_state")
        elif pred == "cooked":
            mt = ed.nonkin[n].get("MaxTemperature")
            if mt is None:
                continue
            thresh = o.param("cookable", "cook_temperature", thr.default_cook_temperature)
            V[key("cooked", [n])] = (np.nan_to_num(mt[:, 0], nan=-1e9) >= thresh, vis(n), "appear", "sim_state")
        elif pred == "frozen":
            tp = ed.nonkin[n].get("Temperature")
            if tp is None:
                continue
            V[key("frozen", [n])] = (np.nan_to_num(tp[:, 0], nan=1e9) <= thr.freeze_temperature, vis(n), "appear", "sim_state")
        elif pred == "on_fire":
            tp = ed.nonkin[n].get("Temperature")
            if tp is None:
                continue
            thresh = o.param("flammable", "ignition_temperature", thr.default_ignition_temperature)
            V[key("on_fire", [n])] = (np.nan_to_num(tp[:, 0], nan=-1e9) >= thresh, vis(n), "appear", "sim_state")

    movables = sorted(n for n in spec.relevant if objs[n].movable)
    inhand_arm = compute_inhand(ed, thr, movables)
    inhand = {h: inhand_arm[(h, "left")] | inhand_arm[(h, "right")] for h in movables}

    def concealed(n):
        """object inside a CLOSED openable container is occluded regardless of the frustum test."""
        c = np.zeros(N, bool)
        for cont, opn in open_of.items():
            if cont != n and objs[cont].has("fillable"):
                c |= ed.inside(n, cont) & ~opn
        return c

    vis_obj = {n: vis(n) & ~concealed(n) for n in movables}
    containers = [n for n in spec.relevant if ed.is_container(n)]

    def inside_any(a, but):
        """a is inside some relevant container other than `but` (a hotdog in the microwave that
        sits on the countertop is not `ontop countertop`)."""
        c = np.zeros(N, bool)
        for cont in containers:
            if cont not in (a, but):  # books stacked inside the same bookcase stay `ontop`
                c |= ed.inside(a, cont) & ~ed.inside(but, cont)
        return c
    held = lambda a: inhand[a] if a in inhand else np.zeros(N, bool)
    for pred, a, b in sorted(universe["binary"]):
        va = vis_obj.get(a, vis(a))
        vb = vis(b)
        if pred == "inside":
            if not ed.is_container(b):
                continue  # `inside <surface>` is never true: keep it out of the key set
            val = ed.inside(a, b)
            v = va & vb & (open_of[b] if b in open_of else np.ones(N, bool))
        elif pred == "ontop":
            val = ed.ontop(a, b) & ~held(a) & ~inside_any(a, b)
            v = va & vb
        elif pred == "nextto":
            val = ed.nextto(a, b)
            v = va & vb
        elif pred == "under":
            val = ed.under(a, b)
            v = va & vb
        elif pred == "touching":
            val = ed.touching(a, b)
            v = va & vb
        elif pred == "attached":
            val = ed.attached(a, b)
            v = va & vb
        else:
            continue
        V[key(pred, [a, b])] = (val, v, "geom", "geom_rule" if pred != "attached" else "sim_state")

    for h in movables:
        onfloor = (ed.aabb(h)[0][:, 2] < thr.floor_z_max) & ed.at_rest(h) & ~inhand[h]
        V[key("onfloor", [h])] = (onfloor, vis_obj[h], "geom", "geom_rule")
        V[key("inhand_left", [h])] = (inhand_arm[(h, "left")], np.ones(N, bool), "robot", "sim_state" if ed.has_ag else "geom_rule")
        V[key("inhand_right", [h])] = (inhand_arm[(h, "right")], np.ones(N, bool), "robot", "sim_state" if ed.has_ag else "geom_rule")
        V[key("inhand", [h])] = (inhand[h], np.ones(N, bool), "robot", "derived")

    # particle-system predicates: not in the raw state -> init value until success terminates
    term = np.where(ed.terminated[:N])[0]
    t_term = int(term[0]) if len(term) else None
    init_val = dict(tg.init_lines)
    goal_val = {}
    for g in tg.goals:
        for pred, args in _goal_literals(g.expr, tg.domains):
            goal_val[key(pred, args)] = True
    for pred, n, sub in sorted(universe["particle"]):
        k = key(pred, [n, sub] if sub else [n])
        base = bool(init_val.get(k, False))
        val = np.full(N, base, bool)
        v = np.zeros(N, bool)
        if t_term is not None and k in goal_val:
            val[t_term:] = goal_val[k]
            v[t_term:] = True
        V[k] = (val, v, "appear", "derived")

    if t_term is not None:
        forced = 0
        for k, want in tg.forced_literals():
            if k not in V:
                continue
            val, v, tag, source = V[k]
            val = np.asarray(val, bool).copy()
            bad = (val[t_term:] != want)
            if bad.any():
                forced += 1
                val[t_term:] = want
                V[k] = (val, v, tag, source)
        ed.decode_stats["goal_forced_keys"] = forced

    base_xy = ed.base_pos[:, :2]
    for tgt in sorted(universe["targets"]):
        if tgt not in ed.pos or tgt == spec.robot:
            continue
        reach = ed.xy_dist_to_aabb(base_xy, tgt) < fitted["reach_dist"]
        visited = np.maximum.accumulate(reach.astype(int)).astype(bool)
        V[key("reachable", [tgt])] = (reach, np.ones(N, bool), "robot", "geom_rule")
        V[key("visited", [tgt])] = (visited, np.ones(N, bool), "robot", "derived")

    frames = np.arange(0, N, 3)  # 10 Hz
    rows = []
    for k, (val, visb, tag, source) in V.items():
        rows.append(pd.DataFrame(dict(
            task=np.int16(spec.task_index), episode=np.int64(ed.ep["raw_episode_id"]),
            frame=frames.astype(np.int32), key=k,
            value=np.asarray(val, bool)[frames], visible=np.asarray(visb, bool)[frames],
            tag=tag, source=source)))
    return pd.concat(rows, ignore_index=True)


def _goal_literals(e, domains):
    """Positive ground atoms a goal conjunct wants true (particle predicates only need this)."""
    from .goal import _atoms
    return _atoms(e, {}, domains)


# ---------------- fitting ----------------
def fit_thresholds(eds: list[EpisodeData], thr: Thresholds) -> tuple[dict, dict]:
    """Data-driven thresholds from the fit episodes."""
    spec = eds[0].spec
    objs = spec.objects
    carry_dists, carry_grips = [], []
    reach_ds = []
    jr_lo, jr_hi, jr_hist = {}, {}, {}
    any_ag = any(ed.has_ag for ed in eds)
    for ed in eds:
        picks = [s for s in ed.segments if s["skill"] == "pick up from" and s["manip"] in ed.pos]
        places = [s for s in ed.segments if s["skill"].startswith("place") and s["manip"]]
        for p in picks:
            nxt = [q for q in places if q["manip"] == p["manip"] and q["start"] >= p["end"]]
            if not nxt:
                continue
            w0, w1 = p["end"], min(nxt[0]["end"], ed.N - 1)
            h = p["manip"]
            for fr in range(w0, w1, 3):
                d = {arm: np.linalg.norm(ed.eef_world[arm][fr] - ed.pos[h][fr]) for arm in ARMS}
                arm = min(d, key=d.get)
                if np.isfinite(d[arm]):
                    carry_dists.append(d[arm])
                    carry_grips.append(ed.gripper_sum[arm][fr])
        for s in ed.segments:
            if s["skill_type"] == "navigation" or not s["objects"]:
                continue
            tgt = s["objects"][-1]
            if tgt not in ed.pos:
                continue
            fr = min(s["start"], ed.N - 1)
            d = ed.xy_dist_to_aabb(ed.base_pos[fr:fr + 1, :2], tgt)[0]
            if np.isfinite(d) and d < 1e8:
                reach_ds.append(d)
        for n in spec.relevant:
            q = ed.joints[n]
            if q.shape[1] == 0:
                continue
            lo, hi = np.nanmin(q, axis=0), np.nanmax(q, axis=0)
            jr_lo[n] = np.minimum(jr_lo.get(n, lo), lo)
            jr_hi[n] = np.maximum(jr_hi.get(n, hi), hi)
            jr_hist.setdefault(n, []).append(np.nanmedian(q, axis=0))
    joint_ranges = {}
    for n in jr_lo:
        med = np.nanmedian(np.stack(jr_hist[n]), axis=0)
        span = jr_hi[n] - jr_lo[n]
        closed_low = np.abs(med - jr_lo[n]) <= np.abs(med - jr_hi[n])
        joint_ranges[n] = dict(lo=jr_lo[n].tolist(), hi=jr_hi[n].tolist(),
                               closed_low=closed_low.tolist(), span=span.tolist())
    if carry_dists and not any_ag and eds[0].proprio is not None:
        grasp_dist = float(np.percentile(carry_dists, 95))
        grip_closed = float(np.percentile(carry_grips, 95))
    else:
        grasp_dist, grip_closed = thr.grasp_dist_default, thr.gripper_closed_default
    reach_p95 = float(np.percentile(reach_ds, 95)) if reach_ds else thr.reach_dist_default
    fitted = dict(
        grasp_dist=grasp_dist, gripper_closed_sum=grip_closed,
        reach_dist=reach_p95 * thr.reach_margin, reach_dist_p95_raw=reach_p95,
        joint_ranges=joint_ranges,
        inhand_source="assisted_grasp" if any_ag else ("geometric" if eds[0].proprio is not None else "none"),
        proprio_source=eds[0].proprio_source,
    )
    notes = dict(
        grasp_dist=("unused: inhand comes from the assisted-grasp constraint in the raw state" if any_ag
                    else "95th pct of min-arm EEF-object distance over carry windows (pick end -> place end)"),
        gripper_closed_sum="95th pct of holding-arm gripper qpos sum over the same windows",
        reach_dist=f"95th pct of base-target horizontal distance at manipulation segment starts x{thr.reach_margin}",
        joint_ranges="observed [min,max] per joint over the fit episodes; closed end = side the median rests on",
        n_carry_samples=len(carry_dists), n_reach_samples=len(reach_ds),
    )
    thr.grasp_dist = grasp_dist
    thr.gripper_closed_sum = grip_closed
    thr.reach_dist = fitted["reach_dist"]
    return fitted, notes
