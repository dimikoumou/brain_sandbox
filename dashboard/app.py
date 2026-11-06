"""Brain Sandbox dashboard.

Start it with:

    streamlit run dashboard/app.py

Everything happens on a trained network loaded from runs/. The sliders build a
stack of perturbations (see brain_sandbox/perturb.py); the network is copied,
changed, and re-tested on the same trials as the intact one.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from brain_sandbox.analysis.behaviour import evaluate_detailed
from brain_sandbox.analysis.pitch_probe import pitch_report
from brain_sandbox.dashboard_logic import default_settings, region_diagram, settings_to_experiment, settings_to_stack
from brain_sandbox.model import to_torch
from brain_sandbox.perturb import describe_stack, perturbed
from brain_sandbox.plotting import GRID, INK, INK_2, MUTED, REGION_COLORS, SURFACE, _style
from brain_sandbox.tasks import TASK_LABELS
from brain_sandbox.train import load_trained

st.set_page_config(page_title="Brain Sandbox", page_icon="🧠", layout="wide")
RUNS = ROOT / "runs"

REGION_HELP = {
    "sensory": "receives everything the network sees and hears",
    "memory": "meant for holding things in mind during a wait",
    "conceptual": "a 'big picture' region; also sends feedback to sensory",
    "executive": "knows which game is being played and gives the answer",
    "flat": "one big unstructured block",
}


# --------------------------------------------------------------------------
# Loading (cached, so it happens once)
# --------------------------------------------------------------------------
def trained_runs():
    if not RUNS.exists():
        return []
    runs = [d for d in sorted(RUNS.iterdir(), reverse=True)
            if (d / "config.yaml").exists() and any(d.glob("seed*/model.pt"))]
    # Show the main model first.
    return sorted(runs, key=lambda d: (0 if "main" in d.name else 1, d.name))


@st.cache_resource(show_spinner="Loading the trained network...")
def load(run: str, seed: int):
    model, tasks, cfg = load_trained(run, seed)
    torch.set_num_threads(2)
    return model, tasks, cfg


@st.cache_data(show_spinner="Testing the intact network...")
def intact_scores(run: str, seed: int, n_trials: int):
    model, tasks, _ = load(run, seed)
    return evaluate_detailed(model, tasks, n_trials)


@st.cache_data(show_spinner="Measuring hidden absolute pitch in the intact network...")
def intact_pitch(run: str, seed: int):
    model, tasks, _ = load(run, seed)
    return pitch_report(model, tasks.layout, tasks.dt_ms, n_reps=12, few_shot=(), include_controls=False)


# --------------------------------------------------------------------------
# Sidebar: choose a network
# --------------------------------------------------------------------------
st.sidebar.title("🧠 Brain Sandbox")
runs = trained_runs()
if not runs:
    st.error("No trained networks found in `runs/`. Train one first:\n\n"
             "`python -m brain_sandbox train --config configs/main_modular.yaml`")
    st.stop()
run = st.sidebar.selectbox("Trained network", runs, format_func=lambda d: d.name)
seeds = sorted(int(p.parent.name[4:]) for p in Path(run).glob("seed*/model.pt"))
seed = st.sidebar.selectbox("Which copy (seed)", seeds,
                            help="Each seed is the same design trained separately from a different random start. "
                                 "Results that hold for only one copy are not trustworthy.")
n_trials = st.sidebar.select_slider("Test trials per game", [64, 128, 256, 512], value=128,
                                    help="More trials = more precise, but slower.")
model, tasks, cfg = load(str(run), seed)
regions = model.module_names
pathways = cfg.model.pathways

if "settings" not in st.session_state or st.session_state.get("settings_for") != str(run):
    st.session_state.settings = default_settings(regions, pathways)
    st.session_state.settings_for = str(run)
s = st.session_state.settings

tab_try, tab_found, tab_how = st.tabs(["Try it yourself", "What we found", "How it works"])

# --------------------------------------------------------------------------
# Tab 1: try it
# --------------------------------------------------------------------------
with tab_try:
    st.markdown("### Break a tiny artificial brain and see what stops working")
    st.markdown(
        "This is a small computer model, trained to play ten simple games with lights and tones. "
        "It is **not** a human brain: it is a toy that helps us think of ideas to test. "
        "Use the controls on the left of this page to damage or tune it, then look at the scores on the right.")

    left, right = st.columns([1, 1.35], gap="large")
    with left:
        st.markdown("#### Your changes")
        if st.button("Reset to the intact network"):
            st.session_state.settings = default_settings(regions, pathways)
            st.rerun()
        for r in regions:
            with st.expander(f"**{r}** — {REGION_HELP.get(r, '')}", expanded=(r == regions[0])):
                s["off"][r] = st.slider("Switch off this % of its units", 0, 100, s["off"][r], 5, key=f"off_{r}")
                s["noise"][r] = st.select_slider("Noise (x normal)", [1.0, 1.5, 2.0, 3.0, 5.0, 10.0], s["noise"][r],
                                                 key=f"noise_{r}")
                s["gain"][r] = st.select_slider("Excitability (x normal)", [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                                                s["gain"][r], key=f"gain_{r}",
                                                help="Like a drug that makes cells more or less responsive.")
                s["tau"][r] = st.select_slider("Speed: time constant (x normal)", [0.5, 1.0, 2.0, 4.0], s["tau"][r],
                                               key=f"tau_{r}", help=">1 makes the region slower to react.")
        if pathways:
            with st.expander("**Cut connections between regions**"):
                for p in pathways:
                    key = f"{p.src}->{p.dst}"
                    name = f"{p.src} → {p.dst}" + ("  (top-down feedback)" if (p.src, p.dst) == ("conceptual", "sensory") else "")
                    s["cut"][key] = st.checkbox(name, s["cut"][key], key=f"cut_{key}")
        with st.expander("**Whole-network knobs**"):
            if cfg.model.dale:
                s["inhibition"] = st.select_slider("Strength of inhibition (x normal)", [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                                                   s["inhibition"], help="Inhibitory cells calm other cells down.")
            s["prune"] = st.slider("Delete the weakest % of connections", 0, 90, s["prune"], 5)

        stack = settings_to_stack(s)
        st.info("**Current change:** " + describe_stack(stack))
        try:
            st.graphviz_chart(region_diagram(regions, pathways, s, cfg.model.input_routes, cfg.model.output_module))
        except Exception:  # graphviz rendering is a nice-to-have
            pass

    with right:
        base = intact_scores(str(run), seed, n_trials)
        changed = evaluate_detailed(perturbed(model, stack), tasks, n_trials) if stack else base
        names = tasks.names
        labels = [TASK_LABELS.get(t, t) for t in names]
        a0 = np.array([base[t]["accuracy"] for t in names])
        a1 = np.array([changed[t]["accuracy"] for t in names])

        st.markdown("#### Scores on each game")
        if stack:
            d = a1 - a0
            worst, best = int(np.argmin(d)), int(np.argmax(d))
            c1, c2 = st.columns(2)
            c1.metric("Most harmed", labels[worst], f"{d[worst]:+.0%}" if d[worst] < -0.005 else "no drop")
            c2.metric("Most improved", labels[best], f"{d[best]:+.0%}" if d[best] > 0.005 else "none")

        fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(names) + 0.9), facecolor=SURFACE)
        _style(ax)
        y = np.arange(len(names))
        ax.barh(y - 0.19, a0, 0.36, color="#c3c2b7", label="intact")
        if stack:
            ax.barh(y + 0.19, a1, 0.36, color="#2a78d6", label="after your change")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of trials done right")
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        if stack:
            for yi, v0, v1 in zip(y, a0, a1):
                if abs(v1 - v0) >= 0.05:
                    ax.text(min(v1, 0.9) + 0.02, yi + 0.19, f"{v1 - v0:+.0%}", va="center", fontsize=7.5, color=INK_2)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if stack:
            imp = [changed[t]["broke_fixation"] for t in names]
            if max(imp) > 0.2:
                t = names[int(np.argmax(imp))]
                st.caption(f"Note: many errors are *impulsive* (answering before the go signal), e.g. "
                           f"{imp[int(np.argmax(imp))]:.0%} of trials in \"{TASK_LABELS.get(t, t)}\".")
        st.caption("This is one trained copy. Small differences (a few %) can be pure chance. "
                   "To check a result properly, save it below and run it on all copies.")

        if tasks.layout.n_tono > 0:
            st.markdown("#### Hidden absolute pitch")
            st.markdown("After a single note has finished playing, can a simple decoder still tell *which* note "
                        "it was (C, C#, D, ...) from each region's activity? Chance is 1 in 12 (8%).")
            if st.checkbox("Measure it (takes about 10–30 seconds)"):
                p0 = intact_pitch(str(run), seed)
                p1 = (pitch_report(perturbed(model, stack), tasks.layout, tasks.dt_ms, n_reps=12, few_shot=(),
                                   include_controls=False) if stack else p0)
                fig, ax = plt.subplots(figsize=(6.4, 2.8), facecolor=SURFACE)
                _style(ax)
                x = np.arange(len(regions))
                ax.bar(x - 0.19, [p0["modules"][r]["post"] for r in regions], 0.36, color="#c3c2b7", label="intact")
                if stack:
                    ax.bar(x + 0.19, [p1["modules"][r]["post"] for r in regions], 0.36, color="#2a78d6",
                           label="after your change")
                ax.axhline(1 / 12, color=INK, ls="--", lw=1)
                ax.text(len(regions) - 0.5, 1 / 12 + 0.02, "chance", fontsize=7.5, color=INK_2, ha="right")
                ax.set_xticks(x)
                ax.set_xticklabels(regions)
                ax.set_ylim(0, 1)
                ax.set_ylabel("note named correctly")
                ax.legend(frameon=False, fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("#### Look inside: activity during one trial")
        task = st.selectbox("Game", names, format_func=lambda t: TASK_LABELS.get(t, t))
        rng = np.random.default_rng(7)
        batch = tasks.generate(task, 1, rng)
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(0)
            _, r0 = model(to_torch(batch.x))
            torch.manual_seed(0)
            _, r1 = perturbed(model, stack)(to_torch(batch.x))
        t = np.arange(batch.x.shape[0]) * tasks.dt_ms
        fig, axes = plt.subplots(len(regions), 2 if stack else 1, figsize=(7, 1.25 * len(regions) + 0.6),
                                 sharex=True, sharey="row", squeeze=False, facecolor=SURFACE)
        for i, rname in enumerate(regions):
            sl = model.module_slices[rname]
            for j, (rates, title) in enumerate([(r0, "intact"), (r1, "after your change")][: 2 if stack else 1]):
                ax = axes[i, j]
                _style(ax)
                ax.plot(t, rates[:, 0, sl].numpy(), lw=0.5, alpha=0.55, color=REGION_COLORS.get(rname, MUTED))
                for ep, (a, _) in batch.epochs.items():
                    ax.axvline(a * tasks.dt_ms, color=GRID, lw=0.8)
                if i == 0:
                    ax.set_title(title, fontsize=9, color=INK)
                if j == 0:
                    ax.set_ylabel(rname, fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("time (ms)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Each line is one unit's activity. Grey vertical lines mark the phases of the trial: "
                   + " → ".join(batch.epochs) + ".")

        st.markdown("#### Save this experiment")
        prediction = st.text_area("Before you test it properly, write what you expect to happen (and why):",
                                  placeholder="e.g. Cutting the feedback will make notes easier to decode in sensory, "
                                              "and the tune game will get worse.")
        if st.button("Save as an experiment file", disabled=not stack):
            path = settings_to_experiment(s, str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
                                          prediction, pitch=tasks.layout.n_tono > 0,
                                          folder=str(ROOT / "configs" / "experiments" / "saved"), root=ROOT)
            st.success(f"Saved to `{path.relative_to(ROOT)}`. Run it on every copy with:\n\n"
                       f"`python -m brain_sandbox perturb --experiment {path.relative_to(ROOT)}`")

# --------------------------------------------------------------------------
# Tab 2: results of the main experiments (pre-computed)
# --------------------------------------------------------------------------
with tab_found:
    findings = ROOT / "docs" / "FINDINGS.md"
    if findings.exists():
        text = findings.read_text()
        # Show images referenced in the markdown with st.image (paths are relative to docs/).
        for block in text.split("\n\n"):
            if block.startswith("![") and "](" in block:
                img = block[block.index("](") + 2: block.rindex(")")]
                st.image(str(ROOT / "docs" / img))
            else:
                st.markdown(block)
    else:
        st.info("The write-up of the main experiments (docs/FINDINGS.md) has not been generated yet.")

# --------------------------------------------------------------------------
# Tab 3: how it works
# --------------------------------------------------------------------------
with tab_how:
    st.markdown((ROOT / "docs" / "HOW_IT_WORKS.md").read_text() if (ROOT / "docs" / "HOW_IT_WORKS.md").exists()
                else "See README.md.")
