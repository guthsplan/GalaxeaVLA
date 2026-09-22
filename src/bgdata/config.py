"""Single source of truth for every threshold used by the label rules.

Values marked fit=True are (re)fitted from the demos at runtime and the fitted
numbers are written to out/<task>/config_fitted.json together with the method.
"""
from dataclasses import dataclass, asdict, field
import json
import pathlib


@dataclass
class Thresholds:
    # --- open (OmniGibson open_state.py rule: joint pos > f * joint range) ---
    open_joint_fraction: float = 0.05          # m.JOINT_THRESHOLD_BY_TYPE (revolute) in open_state.py
    # --- cooked (bddl3 propagated_annots_params.json, hotdog.n.02) ---
    cook_temperature_hotdog: float = 60.0
    # --- inside: AABB containment margins ---
    inside_margin_xy: float = 0.05             # m, expands container AABB horizontally
    inside_margin_z: float = 0.05              # m, expands container AABB vertically
    # --- ontop countertop (fitted: resting z band) ---
    counter_z: float = None                    # fitted per run: median resting z at 'place on' ends
    counter_z_band: float = 0.10               # |z - counter_z| tolerance (m)
    rest_speed: float = 0.12                   # m/s, |lin_vel| below which an object counts as at rest
    # --- inhand (fitted) ---
    grasp_dist: float = None                   # fitted: 95th pct of EEF-object distance while carrying
    gripper_closed_sum: float = None           # fitted: 95th pct of gripper qpos sum while carrying
    inhand_smooth_frames: int = 5              # rolling-median window (10 Hz frames)
    # --- reachable (fitted) ---
    reach_dist: float = None                   # fitted: 95th pct of base-target horizontal distance
                                               # at the first frame of manipulation skills
    reach_margin: float = 1.15                 # multiplicative margin on the p95 (the top ~5% of
                                               # manipulation starts are systematic arrivals, not outliers)
    # --- onfloor ---
    floor_z_max: float = 0.10                  # m, object center height below this counts as on floor
    # --- visibility (geometric proxy; replaces GT-seg pixel count, see README limitations) ---
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
        pathlib.Path(path).write_text(json.dumps(d, indent=2))


# fixed belief constants (must match bg_pi05/belief.py)
PRIOR_CONF = 0.5
PROVISIONAL_CONF = 0.7
DISTURB_FACTOR = 0.8
DISTURB_FLOOR = 0.6
OBS_CONF_TRUE = 0.97
OBS_CONF_FALSE = 0.03
