"""Single source of truth for every threshold used by the label rules.

Values marked fit=True are (re)fitted from the demos at runtime and the fitted
numbers are written to out/<task>/config_fitted.json together with the method.
"""
from dataclasses import dataclass, asdict
import json
import pathlib


@dataclass
class Thresholds:
    # --- open (OmniGibson open_state.py rule: joint pos > f * joint range) ---
    open_joint_fraction: float = 0.05          # m.JOINT_THRESHOLD_BY_TYPE (revolute) in open_state.py
    # --- temperature-derived states (bddl3 propagated_annots_params per synset; defaults below
    #     are OmniGibson's m.DEFAULT_* when the synset has no parameter) ---
    default_cook_temperature: float = 70.0     # object_states/cooked.py
    freeze_temperature: float = 0.0            # object_states/frozen.py
    default_ignition_temperature: float = 250.0  # object_states/on_fire.py
    # --- inside: AABB containment margins ---
    inside_margin_xy: float = 0.05             # m, expands container AABB horizontally
    inside_margin_z: float = 0.05              # m, expands container AABB downward (top is strict)
    container_min_height: float = 0.06         # m, fillable objects lower than this hold no rigid body
    drawer_max_travel: float = 0.7             # joint travel below this (m or rad) is taken as a prismatic drawer
    # --- ontop: object bottom within [top - tol, top + gap] of the support's AABB top ---
    ontop_z_tol: float = 0.03
    ontop_z_gap: float = 0.15
    rest_speed: float = 0.12                   # m/s, |lin_vel| below which an object counts as at rest
    touch_gap: float = 0.02                    # m, AABB gap below which two objects count as touching
    # --- inhand (fitted when no assisted-grasp state exists in the recording) ---
    grasp_dist: float = None                   # fitted: 95th pct of EEF-object distance while carrying
    gripper_closed_sum: float = None           # fitted: 95th pct of gripper qpos sum while carrying
    grasp_dist_default: float = 0.25
    gripper_closed_default: float = 0.08
    inhand_smooth_frames: int = 5              # rolling-median window (10 Hz frames)
    # --- reachable (fitted) ---
    reach_dist: float = None                   # fitted: 95th pct of base-target horizontal distance
    reach_dist_default: float = 1.5
    reach_margin: float = 1.15                 # multiplicative margin on the p95
    # --- onfloor ---
    floor_z_max: float = 0.10                  # m, object bottom below this counts as on floor
    # --- visibility (geometric proxy; replaces GT-seg pixel count) ---
    vis_max_dist: float = 4.0                  # m
    vis_half_fov_deg: float = 45.0             # conservative half field of view per camera
    # --- sampling ---
    fps_in: int = 30
    fps_out: int = 10
    # --- sidecar capacities ---
    sidecar_K: int = 40
    sidecar_R: int = 8

    def save(self, path, notes: dict | None = None):
        d = {"values": asdict(self), "fit_notes": notes or {}}
        pathlib.Path(path).write_text(json.dumps(d, indent=2, default=str))


# fixed belief constants (must match g05.belief_graph.belief)
PRIOR_CONF = 0.5
PROVISIONAL_CONF = 0.7
DISTURB_FACTOR = 0.8
DISTURB_FLOOR = 0.6
OBS_CONF_TRUE = 0.97
OBS_CONF_FALSE = 0.03
