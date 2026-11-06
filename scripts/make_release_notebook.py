"""Build and execute notebooks/release_experiment.ipynb.

    python scripts/make_release_notebook.py --release runs/<..._exp_release> \
        --checkpoint runs/<..._pitch_checkpoint> --text docs/release_interpretation.md

The notebook only LOADS saved results (it does not re-run the experiment), so
it opens fast and always matches the numbers in the report. The written
interpretation comes from a separate markdown file, split into sections by
lines of the form `<!-- section: name -->`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent


def sections(text: str) -> dict[str, str]:
    parts = re.split(r"<!-- section: (\w+) -->\n", text)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def build(release: str, checkpoint: str, text_file: str, out: str) -> Path:
    T = sections(Path(text_file).read_text())
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        md(T["intro"]),
        code(f"""import json, sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from IPython.display import Image, Markdown, display

RELEASE = ROOT / "{release}"
CHECKPOINT = ROOT / "{checkpoint}"
results = json.loads((RELEASE / "results.json").read_text())
print("Experiment folder:", RELEASE.name)
print("Seeds (independently trained networks):", results["seeds"])"""),
        md(T["prediction"]),
        code("""import yaml
spec = yaml.safe_load((RELEASE / "experiment.yaml").read_text())
print(spec["preregistration"])"""),
        md(T["checkpoint"]),
        code("""display(Image(str(CHECKPOINT / "pitch_checkpoint.png")))
display(Markdown((CHECKPOINT / "report.md").read_text().split("![checkpoint](pitch_checkpoint.png)")[1]))"""),
        md(T["results"]),
        code("""display(Image(str(RELEASE / "pitch.png")))
rows = ["| Change | Region | Note named correctly | Change vs intact | Verdict |", "|---|---|---|---|---|"]
for c in results["pitch"]:
    if c["module"] != "sensory":
        continue
    ch = "–" if c["condition"] == "Intact" else f"{c['change']:+.1%} [{c['change_ci_low']:+.1%}, {c['change_ci_high']:+.1%}]"
    rows.append(f"| {c['condition']} | {c['module']} | {c['post']:.1%} | {ch} | {c['verdict']} |")
display(Markdown("\\n".join(rows)))"""),
        md(T["cost"]),
        code("""from brain_sandbox.tasks import TASK_LABELS
display(Image(str(RELEASE / "lesion_matrix.png")))
rows = ["| Change | Pitch game: change in accuracy | Verdict |", "|---|---|---|"]
for c in results["cells"]:
    if c["task"] == "pitchdir":
        rows.append(f"| {c['condition']} | {c['change']:+.0%} [{c['ci_low']:+.0%}, {c['ci_high']:+.0%}] | {c['verdict']} |")
display(Markdown("\\n".join(rows)))"""),
        md(T["verdict"]),
        code("""rows = ["| Prediction | Measure | Predicted | Measured change [95% CI] | Outcome |", "|---|---|---|---|---|"]
for p in results["predictions"]:
    rows.append(f"| {p['condition']} | {p['what']} | {p['direction']} | {p['change']:+.1%} [{p['ci_low']:+.1%}, {p['ci_high']:+.1%}] | **{p['outcome']}** |")
display(Markdown("\\n".join(rows)))"""),
        md(T["discussion"]),
    ]
    nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
    out_path = ROOT / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(out_path.parent)}}).execute()
    nbformat.write(nb, out_path)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="notebooks/release_experiment.ipynb")
    a = ap.parse_args()
    print(build(a.release, a.checkpoint, a.text, a.out))
