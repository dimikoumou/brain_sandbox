"""Plain functions behind the dashboard (kept out of the Streamlit file so they can be tested).

The dashboard describes the user's changes as a simple dict of slider values
("settings"). `settings_to_stack` turns that into a list of perturbation
objects, and `settings_to_experiment` into a YAML experiment file that the
command line can re-run on every seed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from .perturb import (CutPathway, Gain, Inhibition, LesionModule, Noise, Prune, TimeConstant, describe_stack,
                      to_spec)


def default_settings(regions, pathways) -> dict:
    return {
        "off": {r: 0 for r in regions},  # % of units switched off
        "noise": {r: 1.0 for r in regions},  # noise multiplier
        "gain": {r: 1.0 for r in regions},  # excitability multiplier
        "tau": {r: 1.0 for r in regions},  # time-constant multiplier
        "cut": {f"{p.src}->{p.dst}": False for p in pathways},
        "inhibition": 1.0,
        "prune": 0,  # % of weakest connections deleted
    }


def settings_to_stack(s: dict) -> list:
    """Slider values -> list of perturbations (only the ones that change something)."""
    stack = []
    for r, pct in s["off"].items():
        if pct > 0:
            stack.append(LesionModule(r, pct / 100.0))
    for key, cut in s["cut"].items():
        if cut:
            src, dst = key.split("->")
            stack.append(CutPathway(src, dst))
    for r, f in s["noise"].items():
        if f != 1.0:
            stack.append(Noise(f, r))
    for r, f in s["gain"].items():
        if f != 1.0:
            stack.append(Gain(f, r))
    for r, f in s["tau"].items():
        if f != 1.0:
            stack.append(TimeConstant(f, r))
    if s["inhibition"] != 1.0:
        stack.append(Inhibition(s["inhibition"]))
    if s["prune"] > 0:
        stack.append(Prune(s["prune"] / 100.0))
    return stack


def settings_to_experiment(s: dict, run_dir: str, prediction: str, pitch: bool, folder="configs/experiments",
                           root: Path | None = None) -> Path:
    """Write the current changes as an experiment file that the CLI can run on all seeds."""
    stack = settings_to_stack(s)
    label = describe_stack(stack)
    spec = {
        "name": "dashboard_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        "run": str(run_dir),
        "preregistration": prediction or "(no prediction written)",
        "n_trials": 512,
        "pitch": bool(pitch),
        "conditions": [{"label": label, "stack": [to_spec(p) for p in stack]}],
    }
    Path(folder).mkdir(parents=True, exist_ok=True)
    path = Path(folder) / f"{spec['name']}.yaml"
    shown = path.relative_to(root) if root and path.is_relative_to(root) else path
    with open(path, "w") as f:
        f.write("# Saved from the dashboard. Re-run on every seed with:\n")
        f.write(f"#   python -m brain_sandbox perturb --experiment {shown}\n")
        yaml.safe_dump(spec, f, sort_keys=False)
    return path


def region_diagram(regions, pathways, s: dict, input_routes: dict, output_module: str) -> str:
    """Graphviz DOT text: regions as boxes, pathways as arrows; damage shown in red / grey."""
    lines = ["digraph G {", 'rankdir=LR; bgcolor="transparent";',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#c3c2b7"];',
             'edge [color="#52514e", fontname="Helvetica", fontsize=9];',
             'stim [label="sights & sounds", shape=plaintext, style="", fontcolor="#52514e"];',
             'rule [label="which game", shape=plaintext, style="", fontcolor="#52514e"];',
             'resp [label="answer", shape=plaintext, style="", fontcolor="#52514e"];']
    for r in regions:
        off = s["off"].get(r, 0)
        changed = off > 0 or s["noise"].get(r, 1) != 1 or s["gain"].get(r, 1) != 1 or s["tau"].get(r, 1) != 1
        fill = "#f0efec" if off < 100 else "#d6d5d0"
        col = "#e34948" if changed else "#c3c2b7"
        extra = f"\\n{off}% switched off" if off else ""
        lines.append(f'"{r}" [label="{r}{extra}", fillcolor="{fill}", color="{col}", penwidth={2 if changed else 1}];')
    stim_targets = sorted({m for g in ("mod1", "mod2", "tono") for m in input_routes.get(g, [])})
    for m in stim_targets:
        lines.append(f'stim -> "{m}";')
    for m in input_routes.get("rule", []):
        lines.append(f'rule -> "{m}";')
    lines.append(f'"{output_module}" -> resp;')
    for p in pathways:
        cut = s["cut"].get(f"{p.src}->{p.dst}", False)
        style = 'style=dashed, color="#e34948", label="cut"' if cut else ""
        lines.append(f'"{p.src}" -> "{p.dst}" [{style}];')
    lines.append("}")
    return "\n".join(lines)
