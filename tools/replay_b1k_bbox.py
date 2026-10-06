"""Head-camera 2D bboxes for BEHAVIOR 2026 demos by replaying the raw simulator states.

Tasks 50-99 have no 2025 seg_instance_id videos, so behavior_bbox_2025_025fps_v2 has no boxes for them.
This replays a raw episode (task-XXXX/episode_<raw_id>.hdf5) with OmniGibson's HDF5PlaybackWrapper,
restores the recorded state at every frame, renders the head camera (zed_link, 720x720, the
same sensor config as scripts/learning/replay_obs.py) every --stride frames with the `seg_instance`
modality (pixel -> object / particle-system name) and writes the tight box of every target object.

Output (same layout and record format as behavior_bbox_2025_025fps_v2, read by
tools/build_b1k_bbox_trace_sidecars.py):
    <out>/task-XXXX/episode_<raw_id>/frames.jsonl
        {"frame_index", "timestamp", "bbox": {obj: [x1,y1,x2,y2]}, "raw_bbox": {...}, "filtering": {obj: {...}}}
        coordinates normalized to the 720x720 head image, x2/y2 exclusive. The segmentation is exact (no
        video decoding), so "bbox" == "raw_bbox" except that objects with fewer than --min-pixels
        visible pixels are dropped from "bbox".
    <out>/task-XXXX/episode_<raw_id>/report.json

Frame alignment: LeRobot frame i of the 2026 demos is the observation before action i of the last demo
group (the group replay_obs.py replays; its num_samples equals the LeRobot episode length), i.e. the
recorded state[i] -- rendered here after one physics step, as the playback does.

Two box methods (--method):
    proj (default)  project area-weighted surface samples of each target's visual meshes into the head
                    camera and keep the ones in front of the rendered depth (tolerance 2 cm + 3 %). The
                    Replicator segmentation annotators SIGSEGV in the Isaac Sim build of the 2026 challenge
                    image, so this is the method that works; against seg boxes from a run that survived,
                    mean IoU was 0.80.
    seg             tight box of the `seg_instance` pixels (exact, but crashes in that build).

Run it with the OmniGibson python of the BEHAVIOR env, one episode per process, and pin the GPU with
BOTH variables -- the Vulkan renderer ignores CUDA_VISIBLE_DEVICES and follows OMNIGIBSON_GPU_ID:
    CUDA_VISIBLE_DEVICES=2 OMNIGIBSON_GPU_ID=2 OMNIGIBSON_DATA_PATH=/data OMNIGIBSON_HEADLESS=1 \
    python tools/replay_b1k_bbox.py --raw-root <2026-challenge-rawdata> --task 69 --episode 690030 \
        --targets targets.json --out <out_root> --stride 4
A heavy scene needs ~16 GB while loading: one process per 24 GB GPU. scripts/build_bbox_replay.sh in the
solution repo runs whole tasks (targets, workers, retries); tools/downsample_b1k_bbox.py thins the
output to the 0.25 Hz schedule of the 2025 export.
"""
import argparse
import json
import os
import time

import omnigibson as og
import torch as th
from omnigibson.envs import HDF5PlaybackWrapper
from omnigibson.macros import gm
from omnigibson.envs.data_wrapper import (
    _align_scene_object_states_with_recorded_schema,
    _is_system_particle_template_info,
    _is_system_particle_template_name,
)
from omnigibson.systems.macro_particle_system import MacroPhysicalParticleSystem
from omnigibson.utils.python_utils import create_object_from_init_info, h5py_group_to_torch

gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128

IMG = 720


def _scene_inputs(task_name):
    import csv
    import yaml

    meta = os.path.join(gm.DATA_PATH, "2026-challenge-task-instances", "metadata")
    with open(os.path.join(meta, "available_tasks.yaml")) as f:
        scene_model = yaml.safe_load(f)[task_name][0]["scene_model"]
    folder = os.path.join(gm.DATA_PATH, "2026-challenge-task-instances", "scenes", scene_model, "json")
    full_scene_file = next(
        os.path.join(folder, fn) for fn in sorted(os.listdir(folder))
        if task_name in fn and fn.endswith(".json") and "partial_rooms" not in fn
    )
    with open(os.path.join(meta, "B100_task_misc.csv"), newline="", encoding="utf-8") as f:
        rooms = next(row[2].strip().split("\n") for row in csv.reader(f) if task_name in row[1])
    return full_scene_file, rooms


def _boxes(seg, id_to_name, targets, min_pixels):
    raw, kept, filt = {}, {}, {}
    for name in targets:
        ids = [int(k) for k, v in id_to_name.items() if v == name]
        if not ids:
            continue
        mask = th.isin(seg, th.tensor(ids, device=seg.device))
        n = int(mask.sum())
        if n == 0:
            continue
        rows = th.nonzero(mask.any(dim=1)).flatten()
        cols = th.nonzero(mask.any(dim=0)).flatten()
        box = [cols[0].item() / IMG, rows[0].item() / IMG, (cols[-1].item() + 1) / IMG, (rows[-1].item() + 1) / IMG]
        raw[name] = box
        filt[name] = {"pixels": n, "area_threshold": min_pixels}
        if n >= min_pixels:
            kept[name] = box
    return kept, raw, filt


def _mesh_triangles(mesh):
    """(T, 3, 3) triangles of a visual mesh in its local frame, or None."""
    prim = mesh.prim
    if prim.GetPrimTypeInfo().GetTypeName() != "Mesh":
        pts = mesh.points
        if pts is None or len(pts) < 3:
            return None
        import trimesh

        tm = trimesh.convex.convex_hull(pts.numpy())
        return th.tensor(tm.triangles, dtype=th.float32)
    pts = th.tensor(prim.GetAttribute("points").Get(), dtype=th.float32)
    counts = list(prim.GetAttribute("faceVertexCounts").Get())
    idx = list(prim.GetAttribute("faceVertexIndices").Get())
    tris, o = [], 0
    for c in counts:  # fan triangulation
        for j in range(1, c - 1):
            tris.append((idx[o], idx[o + j], idx[o + j + 1]))
        o += c
    if not tris:
        return None
    return pts[th.tensor(tris)]


def _tri_area(t):
    return 0.5 * th.linalg.norm(th.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=1), dim=1)


def _collect_points(scene, targets, max_points, seed=0):
    """name -> [(visual mesh prim, local vertices)] for every target that is an object (systems skipped)."""
    g = th.Generator().manual_seed(seed)
    out = {}
    for name in targets:
        obj = scene.object_registry("name", name)
        if obj is None:
            continue
        meshes = []
        for link in obj.links.values():
            for mesh in link.visual_meshes.values():
                tri = _mesh_triangles(mesh)
                if tri is not None:
                    meshes.append((mesh, tri))
        # area-weighted surface samples (vertices alone are too sparse on large flat faces)
        areas = th.tensor([float(_tri_area(t).sum()) for _, t in meshes]) if meshes else th.zeros(0)
        tot = float(areas.sum())
        samples = []
        for (mesh, tri), a in zip(meshes, areas):
            k = max(16, int(max_points * float(a) / tot)) if tot > 0 else 16
            ar = _tri_area(tri)
            idx = th.multinomial(ar / ar.sum(), k, replacement=True, generator=g) if ar.sum() > 0 else th.randint(len(tri), (k,), generator=g)
            r1, r2 = th.rand(k, generator=g).sqrt(), th.rand(k, generator=g)
            t = tri[idx]
            pts = (1 - r1)[:, None] * t[:, 0] + (r1 * (1 - r2))[:, None] * t[:, 1] + (r1 * r2)[:, None] * t[:, 2]
            samples.append((mesh, pts))
        out[name] = samples
    return out


def _proj_boxes(cam, depth, K, obj_points, min_points):
    """Boxes of the vertices in front of the camera, inside the image and not occluded in @depth."""
    from omnigibson.utils import transform_utils as T

    pos, quat = cam.get_position_orientation()
    R = T.quat2mat(quat)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = depth.shape
    kept, raw, filt = {}, {}, {}
    for name, meshes in obj_points.items():
        pw = th.cat([m.transform_local_points_to_world(p) for m, p in meshes], dim=0)
        pc = (pw - pos) @ R  # world -> camera frame (USD camera: -Z forward, +Y up)
        z = -pc[:, 2]
        front = z > 0.05
        u = cx + fx * pc[:, 0] / z.clamp(min=1e-6)
        v = cy - fy * pc[:, 1] / z.clamp(min=1e-6)
        inside = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if inside.sum() == 0:
            continue
        ui, vi = u[inside].long(), v[inside].long()
        d = depth[vi, ui].float()
        zi = z[inside]
        vis = zi <= d + 0.02 + 0.03 * zi  # not behind the rendered surface (tolerance for vertex vs. pixel)
        n_in, n_vis = int(inside.sum()), int(vis.sum())
        filt[name] = {"in_view": n_in, "visible": n_vis, "min_points": min_points}
        if n_vis == 0:
            continue
        uu, vv = u[inside][vis], v[inside][vis]
        box = [float(uu.min()) / W, float(vv.min()) / H, float(uu.max() + 1) / W, float(vv.max() + 1) / H]
        box = [min(max(b, 0.0), 1.0) for b in box]
        raw[name] = box
        if n_vis >= min_points:
            kept[name] = box
    return kept, raw, filt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--episode", type=int, required=True, help="raw episode id, e.g. 690030")
    ap.add_argument("--targets", required=True, help="json {task: {task_name, targets: [scene object names]}} (see build_bbox_replay.sh)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--min-pixels", type=int, default=32)
    ap.add_argument("--modality", default="seg_instance", help="debug: rgb to test the replay without segmentation")
    ap.add_argument("--no-particle-labels", action="store_true",
                    help="do not give macro particle prims (e.g. dust) semantic labels; the segmentation\n                         render crashes (SIGSEGV in SyntheticData) on task 69 otherwise. Particles then render as unlabelled.")
    ap.add_argument("--method", default="proj", choices=["proj", "seg"],
                    help="proj: project the target objects' visual-mesh vertices into the head camera and keep the "
                         "ones not occluded in the rendered depth (default; the Replicator segmentation annotators "
                         "SIGSEGV in this Isaac Sim build). seg: tight box of seg_instance pixels.")
    ap.add_argument("--max-points", type=int, default=40000, help="proj: surface samples per object (area-weighted)")
    ap.add_argument("--min-points", type=int, default=8, help="proj: visible vertices needed to emit a box")
    ap.add_argument("--max-frames", type=int, default=0, help="debug: stop after this many frames (0 = all)")
    args = ap.parse_args()

    if args.no_particle_labels:
        import omnigibson.systems.macro_particle_system as _mps

        _mps.add_semantic_label = lambda prim, label: None
    spec = json.load(open(args.targets))[str(args.task)]
    task_name, targets = spec["task_name"], spec["targets"]
    ep_dir = os.path.join(args.out, f"task-{args.task:04d}", f"episode_{args.episode:08d}")
    os.makedirs(ep_dir, exist_ok=True)
    input_path = os.path.join(args.raw_root, f"task-{args.task:04d}", f"episode_{args.episode:08d}.hdf5")
    full_scene_file, rooms = _scene_inputs(task_name)
    t0 = time.time()

    gm.ENABLE_TRANSITION_RULES = False
    env = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=input_path,
        output_path=os.path.join("/tmp", f"bbox_replay_{args.episode}.hdf5"),
        full_scene_file=full_scene_file,
        load_room_instances=rooms,
        robot_sensor_config={
            "VisionSensor": {"sensor_kwargs": {"image_height": 480, "image_width": 480}},
            "zed_link:Camera:0": {"sensor_kwargs": {"horizontal_aperture": 40.0, "image_height": IMG, "image_width": IMG}},
        },
        robot_obs_modalities=(["depth_linear"] if args.method == "proj" else args.modality.split(",")),
        include_sensor_names=["zed_link"],
        n_render_iterations=1,
        flush_every_n_steps=0,
        flush_every_n_traj=1,
        include_robot_control=False,
        include_contacts=False,
    )
    t_load = time.time() - t0

    # same episode selection as replay_obs.py: the last demo group
    data_grp = env.input_hdf5["data"]
    demo_ids = sorted(int(k.split("_", 1)[1]) for k in data_grp.keys() if k.startswith("demo_"))
    demo = demo_ids[-1]
    traj = data_grp[f"demo_{demo}"]
    transitions = json.loads(traj.attrs["transitions"])
    traj = h5py_group_to_torch(traj)
    state, state_size, init_metadata = traj["state"], traj["state_size"], traj["init_metadata"]
    n = state.shape[0]

    # --- episode setup, as DataPlaybackWrapper.playback_episode ---
    # The env was just built from this episode's scene file, so the scene.restore() + sim stop/play +
    # env.reset() of playback_episode are only needed when init_metadata carries object attributes.
    # They are skipped otherwise: with a segmentation annotator attached, the first physics step after
    # stop/play crashes the SyntheticData post-process graph (SIGSEGV, deterministic on task 69).
    if len(init_metadata) > 0:
        env.scene.restore(env.scene_file, update_initial_file=True)
        og.sim.stop()
        for i, obj in enumerate(env.scene.objects):
            for attr, vals in init_metadata.items():
                val = vals[i]
                setattr(obj, attr, val.item() if val.ndim == 0 else val)
        og.sim.play()
        env.reset()
    _align_scene_object_states_with_recorded_schema(scene=env.scene, recorded_scene_file=env.recorded_scene_file)
    for robot in env.robots:
        robot.control_enabled = False
    cam = next(s for name, s in env.robots[0].sensors.items() if "zed_link" in name)
    obj_points = _collect_points(env.scene, targets, args.max_points) if args.method == "proj" else None
    K = cam.intrinsic_matrix if args.method == "proj" else None

    og.sim.load_state(state[0, : int(state_size[0])], serialized=True)
    og.sim.step()
    for _ in range(10):  # first render warms up the renderer, as playback does
        og.sim.render()

    n_boxes, frames_out = 0, []
    with open(os.path.join(ep_dir, "frames.jsonl"), "w") as fout:
        for i in range(min(n, args.max_frames) if args.max_frames else n):
            og.sim.load_state(state[i, : int(state_size[i])], serialized=True)
            for obj in env.scene.objects:
                obj.keep_still()
            for system in env.scene.systems:
                if isinstance(system, MacroPhysicalParticleSystem):
                    system.set_particles_velocities(
                        lin_vels=th.zeros((system.n_particles, 3)), ang_vels=th.zeros((system.n_particles, 3))
                    )
            if i % args.stride == 0:
                og.sim.step()
                og.sim.render()
                obs, info = cam.get_obs()
                if args.method == "proj":
                    kept, raw, filt = _proj_boxes(cam, obs["depth_linear"], K, obj_points, args.min_points)
                elif "seg_instance" not in args.modality.split(","):
                    kept, raw, filt = {}, {}, {}
                else:
                    kept, raw, filt = _boxes(obs["seg_instance"], info["seg_instance"], targets, args.min_pixels)
                n_boxes += len(kept)
                frames_out.append(i)
                fout.write(json.dumps({"frame_index": i, "timestamp": i / 30.0, "bbox": kept,
                                       "raw_bbox": raw, "filtering": filt}) + "\n")
            # replay recorded object additions/removals so the next serialized state matches the scene
            if str(i) in transitions:
                cur = transitions[str(i)]
                scene = og.sim.scenes[0]
                added, removed = set(cur["systems"]["add"]), set(cur["systems"]["remove"])
                for s in cur["systems"]["add"]:
                    scene.get_system(s, force_init=True)
                for s in cur["systems"]["remove"]:
                    scene.clear_system(s)
                for name in cur["objects"]["remove"]:
                    if _is_system_particle_template_name(name, removed):
                        continue
                    scene.remove_object(scene.object_registry("name", name))
                for j, info_add in enumerate(cur["objects"]["add"]):
                    if _is_system_particle_template_info(info_add, added):
                        continue
                    obj = create_object_from_init_info(info_add)
                    scene.add_object(obj)
                    obj.set_position(th.ones(3) * 100.0 + th.ones(3) * 5 * j)
                og.sim.step()
            if i % 500 == 0:
                print(f"[bbox] ep {args.episode} frame {i}/{n} boxes {n_boxes}", flush=True)

    report = {
        "task_id": args.task, "raw_episode_id": args.episode, "demo_group": demo,
        "length_2025": n, "length_2026": n,  # the sidecar builder checks these against the LeRobot length
        "complete": True, "packable": True, "output_fps": 30.0 / args.stride, "source_fps": 30.0,
        "stride": args.stride, "sampled_frames": len(frames_out), "boxes": n_boxes, "targets": targets,
        "min_pixels": args.min_pixels, "min_points": args.min_points, "max_points": args.max_points, "image_size": [IMG, IMG],
        "decoder_version": f"sim_replay_{args.method}_v1",
        "decoder": ("OmniGibson HDF5 state replay, head zed_link 720x720; box of area-weighted surface samples of the target visual meshes that are "
                    "in view and not behind the rendered depth (tol 2 cm + 3 %)") if args.method == "proj" else
                   "OmniGibson HDF5 state replay, head zed_link 720x720 seg_instance, tight box of visible pixels",
        "objects_projected": sorted(obj_points) if obj_points else None,
        "load_s": round(t_load, 1), "elapsed_s": round(time.time() - t0, 1),
    }
    json.dump(report, open(os.path.join(ep_dir, "report.json"), "w"), indent=1)
    print(f"[bbox] done ep {args.episode}: {len(frames_out)} frames, {n_boxes} boxes, {report['elapsed_s']} s", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
