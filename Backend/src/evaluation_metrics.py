import os
import json
import numpy as np
import matplotlib.pyplot as plt


def extract_archive_map(archive):
    """Return {cell_index_tuple: objective} for filled cells in a pyribs GridArchive.
    
    PyRibs returns a linear (flat) index. We convert it to multi-dimensional
    index using the archive's dims via np.unravel_index.
    """
    m = {}
    dims = tuple(int(d) for d in archive.dims)  # Get archive dimensions
    
    for rec in archive:
        # Get index (either attribute or dict key)
        idx = getattr(rec, "index", None)
        if idx is None and hasattr(rec, "get"):
            idx = rec.get("index", None)
        
        # Get objective
        obj = getattr(rec, "objective", None)
        if obj is None:
            obj = getattr(rec, "value", None)
        if obj is None and hasattr(rec, "get"):
            obj = rec.get("objective", rec.get("value", None))
        
        if idx is None or obj is None:
            continue
        
        # Convert linear index to multi-dimensional index tuple
        try:
            linear_idx = int(idx)
            # Use 'C' order (row-major) to match PyRibs internal indexing
            idx_tuple = tuple(int(v) for v in np.unravel_index(linear_idx, dims))
        except Exception:
            # Fallback: maybe already a tuple/array
            try:
                idx_tuple = tuple(int(v) for v in np.atleast_1d(idx).tolist())
            except Exception:
                idx_tuple = (int(idx),)
        
        m[idx_tuple] = float(obj)
    return m


def union_max(run_maps):
    """Compute M_xy = max across runs for each cell, from a list of per-run maps."""
    M = {}
    for m in run_maps:
        for cell, val in m.items():
            M[cell] = max(M.get(cell, -np.inf), val)
    # Include all finite filled cells regardless of sign
    return {cell: v for cell, v in M.items() if np.isfinite(v)}


def global_reliability(m, M):
    """G(m) = mean over union cells of (m_xy / M_xy), missing m_xy -> 0."""
    if not M:
        return 0.0
    ratios = []
    for cell, Mxy in M.items():
        mxy = m.get(cell, 0.0)
        ratios.append(mxy / Mxy if Mxy > 0 else 0.0)
    return float(np.mean(ratios)) if ratios else 0.0


def precision_opt_in(m, M):
    """P(m) = mean over cells filled by run m of (m_xy / M_xy)."""
    if not m:
        return 0.0
    ratios = []
    for cell, mxy in m.items():
        Mxy = M.get(cell, None)
        if Mxy is None or Mxy <= 0:
            continue
        ratios.append(mxy / Mxy)
    return float(np.mean(ratios)) if ratios else 0.0


def coverage_relative(m, FM_count):
    """C(m) = n(m) / n(FM), where FM_count is # union-filled cells across all runs."""
    n_m = len(m)
    return float(n_m) / float(FM_count) if FM_count > 0 else 0.0


def coverage_absolute(m, dims):
    """Absolute coverage vs archive capacity."""
    capacity = int(np.prod(dims))
    return float(len(m)) / float(capacity) if capacity > 0 else 0.0


def qd_score(m):
    """QD-Score = sum of objective values over filled cells."""
    return float(np.sum(list(m.values()))) if m else 0.0


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv(rows, cols, path):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def plot_series(xs, ys_dict, title, xlabel, ylabel, out_path):
    plt.figure(figsize=(8, 5))
    for label, ys in ys_dict.items():
        plt.plot(xs, ys, label=label, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_bar(labels, values, title, ylabel, out_path):
    plt.figure(figsize=(8, 5))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_heatmap_2d(m, dims, title, out_path):
    """Plot 2D heatmap by projecting max over the third dim (rolloff) to centroid×flux."""
    H, W, D = dims
    data = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            vals = []
            for k in range(D):
                v = m.get((i, j, k), None)
                if v is not None:
                    vals.append(v)
            data[i, j] = max(vals) if vals else 0.0
    plt.figure(figsize=(6, 5))
    plt.imshow(data.T, origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(label="Objective")
    plt.title(title)
    plt.xlabel("Centroid bin")
    plt.ylabel("Spectral Flux bin")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
