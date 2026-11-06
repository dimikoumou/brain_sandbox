"""Emergent clusters of units, following Yang et al. 2019.

1. Task variance: for each unit and each task, how much does the unit's
   activity vary from trial to trial during the task? A unit that does not
   care about a task has ~0 variance on it.
2. Normalise each unit's task-variance profile by its maximum, so a unit is
   described by *which* tasks it cares about, not by how active it is.
3. Cluster units with similar profiles (k-means; the number of clusters k is
   chosen by the silhouette score).

Then we can ask: do the clusters line up with the regions we designed, or do
they cut across them? And we can lesion a cluster like a region.

Cluster numbers are arbitrary and differ between seeds. To compare clusters
across seeds, `match_to_reference` relabels each seed's clusters to the most
similar cluster of a reference seed (by correlation of their average task
profiles, with an optimal one-to-one assignment).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..model import to_torch
from .behaviour import TEST_SEED


@torch.no_grad()
def task_variance(model, tasks, n_trials: int = 256, batch_size: int = 64, seed: int = TEST_SEED) -> np.ndarray:
    """(N units, n tasks) array of trial-to-trial variance, noise switched off."""
    V = np.zeros((model.N, len(tasks.names)))
    for k, name in enumerate(tasks.names):
        rng = np.random.default_rng([seed, k])
        parts = []
        for _ in range(max(1, n_trials // batch_size)):
            b = tasks.generate(name, batch_size, rng)
            _, rates = model(to_torch(b.x), noise=False)
            start = b.epochs["fixation"][1]  # from the first stimulus on
            parts.append(rates[start:].var(dim=1).mean(dim=0).numpy())
        V[:, k] = np.mean(parts, axis=0)
    return V


def active_units(V: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    total = V.sum(1)
    return total > threshold * total.max()


def normalise(V: np.ndarray) -> np.ndarray:
    return V / np.maximum(V.max(1, keepdims=True), 1e-12)


def cluster_units(V: np.ndarray, k: int | None = None, k_range=range(3, 13), seed: int = 0) -> dict:
    """Cluster active units by their normalised task-variance profiles.

    Returns labels (N,), with -1 for inactive units, the chosen k, and the
    silhouette score for each k tried.
    """
    act = active_units(V)
    X = normalise(V[act])
    scores = {}
    ks = [k] if k is not None else [kk for kk in k_range if kk < len(X)]
    best = None
    for kk in ks:
        km = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(X)
        s = silhouette_score(X, km.labels_)
        scores[kk] = float(s)
        if best is None or s > best[0]:
            best = (s, kk, km.labels_)
    labels = np.full(V.shape[0], -1)
    labels[act] = best[2]
    labels = _order_by_peak_task(labels, normalise(V))
    return {"labels": labels, "k": best[1], "silhouette": scores, "active": act}


def _order_by_peak_task(labels, Vn):
    """Renumber clusters so their order follows the task they care about most (nicer plots)."""
    ids = [c for c in np.unique(labels) if c >= 0]
    peak = {c: (np.argmax(Vn[labels == c].mean(0)), -Vn[labels == c].mean(0).max()) for c in ids}
    order = sorted(ids, key=lambda c: peak[c])
    new = labels.copy()
    for i, c in enumerate(order):
        new[labels == c] = i
    return new


def centroids(V: np.ndarray, labels: np.ndarray) -> np.ndarray:
    Vn = normalise(V)
    return np.stack([Vn[labels == c].mean(0) for c in range(labels.max() + 1)])


def match_to_reference(ref_centroids: np.ndarray, centroids_: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Relabel clusters so that cluster i is the one most like reference cluster i."""
    k = min(len(ref_centroids), len(centroids_))
    sim = np.corrcoef(np.vstack([ref_centroids, centroids_]))[: len(ref_centroids), len(ref_centroids):]
    rows, cols = linear_sum_assignment(-sim)
    new = np.full_like(labels, -1)
    for r, c in zip(rows, cols):
        new[labels == c] = r
    # Clusters with no partner (if k differs) keep a fresh number.
    nxt = len(ref_centroids)
    for c in range(len(centroids_)):
        if c not in cols:
            new[labels == c] = nxt
            nxt += 1
    return new


def region_composition(labels: np.ndarray, module_slices: dict) -> np.ndarray:
    """(n clusters, n regions) count of each region's units in each cluster."""
    n = labels.max() + 1
    comp = np.zeros((n, len(module_slices)), dtype=int)
    for j, sl in enumerate(module_slices.values()):
        lab = labels[sl]
        for c in range(n):
            comp[c, j] = int((lab == c).sum())
    return comp


def describe_cluster(centroid: np.ndarray, task_names, labels=None, threshold: float = 0.5, max_tasks: int = 3) -> str:
    """Short description: the (at most 3) tasks this cluster is most busy with, by short task name."""
    busy = [task_names[i] for i in np.argsort(-centroid) if centroid[i] >= threshold]
    if not busy:
        busy = [task_names[int(np.argmax(centroid))]]
    extra = f" +{len(busy) - max_tasks}" if len(busy) > max_tasks else ""
    return "mostly " + ", ".join(busy[:max_tasks]) + extra
