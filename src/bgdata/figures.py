"""Belief-graph snapshot figures + confidence timeline for one episode."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .operators import parse_key

NODE_POS = {
    "fridge_dszchb_0": (0.12, 0.62),
    "countertop_kelker_0": (0.5, 0.80),
    "microwave_abzvij_0": (0.88, 0.62),
    "hotdog_207": (0.34, 0.38),
    "hotdog_208": (0.66, 0.38),
    "robot_r1": (0.5, 0.10),
}
SHORT = {"fridge_dszchb_0": "fridge", "countertop_kelker_0": "countertop",
         "microwave_abzvij_0": "microwave", "hotdog_207": "hotdog_207",
         "hotdog_208": "hotdog_208", "robot_r1": "robot"}
REL_PREDS = ("inside", "ontop", "inhand", "onfloor")
ATTR_PREDS = ("open", "toggled_on", "cooked")


def draw_graph(rec: dict, path: str, title_extra: str = ""):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(f"step {rec['step']} — {rec['skill']} {title_extra}\n"
                 f"progress {rec['progress']:.2f}   remaining: {rec['remaining_goal_lines']}",
                 fontsize=10)
    attrs = {n: [] for n in NODE_POS}
    edges = []
    for key, (val, p, obs, src) in rec["predicates"].items():
        name, args = parse_key(key)
        if name in ("inhand_left", "inhand_right"):
            continue
        if name in REL_PREDS and val:
            a = args[0]
            b = args[1] if len(args) > 1 else "robot_r1"
            if name == "onfloor":
                attrs.setdefault(a, []).append((f"onfloor", p, obs, src))
                continue
            if a in NODE_POS and b in NODE_POS:
                edges.append((a, b, name, p, obs, src))
        elif name in ATTR_PREDS:
            o = args[0]
            label = name if val else ("closed" if name == "open" else f"not {name}")
            conf = p if val else 1.0 - p  # confidence of the displayed label
            attrs.setdefault(o, []).append((label, conf, obs, src))
        elif name in ("reachable", "visited") and val:
            attrs.setdefault(args[0], []).append((name, p, obs, src))
    for n, (x, y) in NODE_POS.items():
        ax.scatter([x], [y], s=1800, c="#dbe9f6" if "hotdog" not in n else "#fde6cf",
                   edgecolors="#333", zorder=2)
        ax.text(x, y, SHORT[n], ha="center", va="center", fontsize=9, zorder=3)
        lines = [f"{lab} ({p:.2f}{'*' if obs else ''})" for lab, p, obs, src in attrs.get(n, [])]
        if lines:
            ax.text(x, y - 0.075, "\n".join(lines), ha="center", va="top", fontsize=7,
                    color="#444", zorder=3)
    for a, b, name, p, obs, src in edges:
        xa, ya = NODE_POS[a]; xb, yb = NODE_POS[b]
        style = "-" if obs else "--"
        color = "#1b6f2f" if obs else "#888"
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="->", linestyle=style, color=color, lw=1.6,
                                    shrinkA=28, shrinkB=28), zorder=1)
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        ax.text(mx, my + 0.02, f"{name} {p:.2f}", ha="center", fontsize=8,
                color=color, zorder=3)
    ax.text(0.01, 0.01, "solid = observed this frame · dashed = memory · (p) = P(true) · * = observed",
            fontsize=7, color="#666")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def draw_timeline(records: list[dict], segments: list[dict], keys: list[str], path: str):
    steps = [r["step"] for r in records]
    fig, ax = plt.subplots(figsize=(13, 5))
    for key in keys:
        ps = [r["predicates"].get(key, [None, np.nan, None, None])[1] for r in records]
        ax.plot(steps, ps, label=key, lw=1.5)
    for s in segments:
        ax.axvline(s["start"], color="#bbb", lw=0.6, zorder=0)
        ax.text(s["start"], 1.03, s["skill"], rotation=60, fontsize=6, ha="left", va="bottom")
    ax.set_xlabel("frame (30 fps index, 10 Hz sampling)")
    ax.set_ylabel("P(true)")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
