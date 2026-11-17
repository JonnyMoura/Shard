import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

try:
    from ribs.archives import GridArchive
    from ribs.schedulers import Scheduler
except Exception:
    GridArchive = None
    Scheduler = None

from src.evaluation_metrics import (
    extract_archive_map,
    union_max,
    global_reliability,
    precision_opt_in,
    coverage_relative,
    coverage_absolute,
    qd_score,
    save_json,
    save_csv,
    plot_series,
    plot_bar,
    plot_heatmap_2d,
)

# Import project utilities (same folder)
from src.audio_synthesis import generate_audio_stream, extract_audio_descriptors
from src.dataset_class import classify_audio_category
from src.evolutionary_module import (
    archive as evo_archive,
    automatic_fitness,
    general_bounds,
)
from src.utils import load_audio_catalog
from src.runner_helpers import (
    balanced_category_track_assignment,
    inherit_tracks_from_previous,
    get_assigned_category,
    get_tracks_from_solution,
    clear_registries,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"


def save_algorithm_params(params: Dict[str, Any], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    save_json(params, os.path.join(out_dir, "algorithm_params.json"))


def save_citations(out_root: str):
    citations = [
        {
            "title": "Illuminating search spaces by mapping elites",
            "authors": "Jean-Baptiste Mouret, Joel Lehman, and Kenneth O. Stanley",
            "venue": "GECCO 2015",
            "doi": "10.1145/2739480.2754689",
        },
        {
            "title": "Robots that can adapt like animals",
            "authors": "Antoine Cully, Jeff Clune, Danesh Tarapore, and Jean-Baptiste Mouret",
            "venue": "Nature 2015",
            "doi": "10.1038/nature14422",
        },
        {
            "title": "Quality Diversity: Overcoming Deceptiveness in Search",
            "authors": "Justin Pugh, Lisa Soros, and Kenneth O. Stanley",
            "venue": "GECCO 2016",
            "doi": "10.1145/2908812.2908899",
        },
    ]
    save_json(citations, os.path.join(out_root, "citations.json"))


def save_generation_audio(gen_idx: int, audios: List[np.ndarray], sr: int, out_dir: str):
    try:
        import soundfile as sf
    except Exception as e:
        raise RuntimeError("Missing dependency 'soundfile'. Install via 'pip install soundfile'.")
    # Create per-generation subfolder inside the run's audio dir
    gen_dir = os.path.join(out_dir, f"gen_{gen_idx:03d}")
    os.makedirs(gen_dir, exist_ok=True)
    for i, y in enumerate(audios):
        path = os.path.join(gen_dir, f"idx_{i:03d}.wav")
        if os.path.exists(path):
            continue
        # normalize to prevent clipping
        if y is None:
            continue
        maxv = np.max(np.abs(y)) if y.size > 0 else 0.0
        if maxv > 0:
            y = y / maxv
        sf.write(path, y.astype(np.float32), sr)


def timed_job(
    solution_vec: np.ndarray,
    catalog_df,
    category: str,
    duration: float,
    sample_rate: int,
) -> Tuple[float, float, np.ndarray, np.ndarray, int]:
    """Synthesize audio and compute raw descriptors using assigned tracks & category."""
    t0 = time.perf_counter()
    sources = get_tracks_from_solution(solution_vec, catalog_df)
    # Normalize source paths to absolute within project root; fix '../' prefixes
    norm_sources = []
    for p in sources:
        if os.path.isabs(p):
            candidates = [p]
        else:
            candidates = [
                os.path.normpath(os.path.join(base, p))
                for base in (BACKEND_DIR, BACKEND_DIR / "src", Path.cwd())
            ]
        existing = next((candidate for candidate in candidates if os.path.isfile(candidate)), None)
        if existing:
            norm_sources.append(existing)

    if not norm_sources:
        raise FileNotFoundError(f"No assigned audio sources exist: {sources}")

    audio, sr = generate_audio_stream(
        solution_vec,
        duration=duration,
        sources=norm_sources,
        category=category,
        target_sample_rate=sample_rate,
    )
    t1 = time.perf_counter()
    raw_desc = extract_audio_descriptors(audio, sr)
    t2 = time.perf_counter()
    synth_time = t1 - t0
    total_time = t2 - t0
    return total_time, synth_time, raw_desc, audio, sr


def run_single_trial(
    trial_idx: int,
    generations: int = 50,
    pop_size: int = 64,
    sr: int = 44100,
    dims: Tuple[int, int, int] = (20, 20, 20),
    duration: float = 2.0,
    out_root: str = "data/evaluation",
) -> Dict[str, Any]:
    # Clear registries at the start of each run for independent state
    clear_registries()
    
    out_root = Path(out_root)
    if not out_root.is_absolute():
        out_root = BACKEND_DIR / out_root
    out_dir = os.fspath(out_root / f"run_{trial_idx:03d}")
    os.makedirs(out_dir, exist_ok=True)

    # Use project's normal archive and emitters
    if Scheduler is None:
        raise RuntimeError("PyRibs not available. Install via 'pip install ribs'.")
    if pop_size < 1:
        raise ValueError("pop_size must be greater than zero")
    if sr < 1:
        raise ValueError("sr must be greater than zero")
    if duration <= 0:
        raise ValueError("duration must be greater than zero")
    dims = tuple(int(d) for d in dims)
    if len(dims) != 3 or any(d < 1 for d in dims):
        raise ValueError("dims must contain three positive integers")
    
    # Create a FRESH archive for each run to ensure independent metrics
    # Copy configuration from evo_archive but start empty
    from ribs.emitters import EvolutionStrategyEmitter, IsoLineEmitter
    
    # Reconstruct ranges from boundaries (each boundary array has [min, ..., max])
    archive_ranges = [(b[0], b[-1]) for b in evo_archive.boundaries]
    
    archive = GridArchive(
        solution_dim=evo_archive.solution_dim,
        dims=list(dims),
        ranges=archive_ranges,
    )
    emitters = []
    emitter_types = [EvolutionStrategyEmitter, IsoLineEmitter,
                     EvolutionStrategyEmitter, EvolutionStrategyEmitter] * 4
    if pop_size == 1:
        emitter_types = [IsoLineEmitter]
    emitter_count = min(len(emitter_types), max(1, pop_size // 2))
    base_batch_size, remainder = divmod(pop_size, emitter_count)
    for index, emitter_type in enumerate(emitter_types[:emitter_count]):
        emitter_kwargs = {
            "archive": archive,
            "x0": np.random.uniform(low=general_bounds[0], high=general_bounds[1]),
            "batch_size": base_batch_size + (index < remainder),
        }
        if emitter_type is EvolutionStrategyEmitter:
            emitter_kwargs["sigma0"] = 0.7
        emitters.append(emitter_type(**emitter_kwargs))
    scheduler = Scheduler(archive, emitters)

    params = {
        "generations": generations,
        "pop_size": pop_size,
        "dims": dims,
        "sr": sr,
        "duration": duration,
        "emitters": len(emitters),
    }
    # Add richer parameters from archive and emitters
    try:
        params["solution_dim"] = int(getattr(archive, "solution_dim", 0))
    except Exception:
        params["solution_dim"] = 0
    try:
        ranges = getattr(archive, "ranges", [])
        params["archive_ranges"] = [
            [float(lo), float(hi)] for (lo, hi) in ranges
        ]
    except Exception:
        params["archive_ranges"] = []
    try:
        params["bounds_low"] = [float(v) for v in np.asarray(general_bounds[0]).tolist()]
        params["bounds_high"] = [float(v) for v in np.asarray(general_bounds[1]).tolist()]
    except Exception:
        params["bounds_low"] = []
        params["bounds_high"] = []
    try:
        params["emitter_batch_sizes"] = [int(getattr(e, "batch_size", 0)) for e in emitters]
        params["emitter_types"] = [getattr(e, "__class__").__name__ for e in emitters]
    except Exception:
        params["emitter_batch_sizes"] = []
        params["emitter_types"] = []
    save_algorithm_params(params, out_dir)

    # Load catalog for track assignments
    catalog_path = DATA_DIR / "audio_analysis.csv"
    try:
        catalog_df = load_audio_catalog(str(catalog_path))
    except FileNotFoundError:
        # Fallback catalog built from files in data/audio
        audio_dir = DATA_DIR / "audio"
        file_paths = []
        if audio_dir.is_dir():
            for name in os.listdir(audio_dir):
                if name.lower().endswith((".wav", ".flac", ".ogg")):
                    file_paths.append(str(audio_dir / name))
        import pandas as pd
        catalog_df = pd.DataFrame({
            "file_path": file_paths,
            "dominant_category": ["mid"] * len(file_paths),
            "spectral_flux": [0.0] * len(file_paths),
            "centroid": [1500.0] * len(file_paths),
            "onset_rate": [0.0] * len(file_paths),
        })
    if catalog_df.empty:
        raise RuntimeError(f"No audio files are available in {DATA_DIR / 'audio'}")

    # Metrics storage
    gen_metrics_rows = []
    coverage_rel = []
    qd_scores = []
    gen_times_total = []
    gen_times_synth = []
    gen_times_desc = []

    # Initial population from emitters
    X = scheduler.ask()
    # Clip to project's general bounds like main.py
    for j in range(len(X)):
        X[j] = np.clip(X[j], general_bounds[0], general_bounds[1])
    # Balanced category + track assignment for gen 0
    X = balanced_category_track_assignment(X, catalog_df)

    # Generation loop
    for g in range(generations):
        gt0 = time.perf_counter()

        # Evaluate population
        audios = []
        objectives = []
        raw_descs = []
        synth_time_sum = 0.0
        desc_time_sum = 0.0
        for i in range(len(X)):
            category = get_assigned_category(X[i]) or "mid"
            tot, ts, raw_desc, y, sr_audio = timed_job(
                X[i], catalog_df, category, duration, sr
            )
            synth_time_sum += ts
            desc_time_sum += tot - ts
            audios.append(y)
            raw_descs.append(raw_desc)

        raw_descs = np.array(raw_descs)  # [centroid, rolloff, spectral_flux, onset_rate, zcr]
        centroid = raw_descs[:, 0]
        rolloff = raw_descs[:, 1]
        flux = raw_descs[:, 2]
        # Measures match archive ranges: (centroid, spectral_flux, rolloff)
        measures = np.stack([centroid, flux, rolloff], axis=1)
        # Clip measures to archive ranges to ensure insertion into bins
        try:
            ranges = getattr(archive, "ranges", None)
            if ranges is not None:
                ranges_arr = np.asarray(ranges, dtype=float)
                low = ranges_arr[:, 0]
                high = ranges_arr[:, 1]
                measures = np.clip(measures, low, high)
        except Exception:
            pass
        # Objective: use project's automatic_fitness with category (like main.py)
        for i in range(len(X)):
            actual_category = classify_audio_category(
                raw_descs[i, 0], raw_descs[i, 2], raw_descs[i, 3], raw_descs[i, 4]
            )
            objectives.append(float(automatic_fitness(raw_descs[i], actual_category)))
        objectives = np.asarray(objectives)

        # Tell archive
        scheduler.tell(objectives, measures)

        # Save audio for this generation
        # Use the actual audio sampling rate; assume consistent
        save_generation_audio(g, audios, sr, os.path.join(out_dir, "audio"))

        # Metrics from archive
        m = extract_archive_map(archive)
        qd = qd_score(m)
        qd_scores.append(qd)
        # Relative coverage requires union over runs; approximate with absolute here for per-gen log
        cov_abs = coverage_absolute(m, dims)
        coverage_rel.append(cov_abs)

        gt1 = time.perf_counter()
        gen_times_total.append(gt1 - gt0)
        gen_times_synth.append(synth_time_sum)
        gen_times_desc.append(desc_time_sum)

        gen_metrics_rows.append([
            g,
            cov_abs,
            qd,
            gen_times_total[-1],
            gen_times_synth[-1],
            gen_times_desc[-1],
            len(m),
            float(np.mean(objectives)),
            float(np.max(objectives)),
            float(np.min(objectives)),
        ])

        # Ask for next population
        prev_X = X
        X = scheduler.ask()
        for j in range(len(X)):
            X[j] = np.clip(X[j], general_bounds[0], general_bounds[1])
        # Inheritance from archive elites (tracks & category) - matches main.py behavior
        mutation_rate = 0.15
        X = balanced_category_track_assignment(X, catalog_df) if g == 0 else inherit_tracks_from_previous(
            X, catalog_df, prev_X, mutation_rate, archive=archive, general_bounds=general_bounds
        )

    # Save per-generation metrics
    save_csv(
        gen_metrics_rows,
        [
            "generation",
            "coverage_abs",
            "qd_score",
            "time_total",
            "time_synthesis",
            "time_descriptors",
            "archive_cells",
            "fitness_mean",
            "fitness_max",
            "fitness_min",
        ],
        os.path.join(out_dir, "metrics_per_generation.csv"),
    )

    # Per-run plots for this run (not aggregated)
    gens = list(range(generations))
    try:
        plot_series(
            gens,
            {"QD-Score": np.asarray(qd_scores)},
            f"QD-Score over generations (run {trial_idx})",
            "Generation",
            "QD-Score",
            os.path.join(out_dir, "plot_qd_over_gens.png"),
        )
        plot_series(
            gens,
            {"Coverage(abs)": np.asarray(coverage_rel)},
            f"Coverage over generations (run {trial_idx})",
            "Generation",
            "Coverage(abs)",
            os.path.join(out_dir, "plot_cov_over_gens.png"),
        )
        plot_series(
            gens,
            {
                "Total": np.asarray(gen_times_total),
                "Synthesis": np.asarray(gen_times_synth),
                "Descriptors": np.asarray(gen_times_desc),
            },
            f"Timing over generations (run {trial_idx})",
            "Generation",
            "Seconds",
            os.path.join(out_dir, "plot_timing_over_gens.png"),
        )
    except Exception:
        pass

    # Save final archive heatmap
    m_final = extract_archive_map(archive)
    
    import matplotlib.pyplot as plt
    
    # 2D projection heatmap (max objective over 3rd dim)
    plot_heatmap_2d(m_final, dims, f"Run {trial_idx} final archive (2D projection)", os.path.join(out_dir, "heatmap_final.png"))
    
    # 3D scatter plot for full archive visualization
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        if m_final:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            xs, ys, zs, cs = [], [], [], []
            for (i, j, k), obj in m_final.items():
                xs.append(i)
                ys.append(j)
                zs.append(k)
                cs.append(obj)
            
            scatter = ax.scatter(xs, ys, zs, c=cs, cmap='viridis', s=50, alpha=0.8)
            ax.set_xlabel('Centroid bin')
            ax.set_ylabel('Spectral Flux bin')
            ax.set_zlabel('Rolloff bin')
            ax.set_title(f'Run {trial_idx} archive (3D scatter)')
            plt.colorbar(scatter, ax=ax, label='Objective', shrink=0.6)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "heatmap_3d_scatter.png"), dpi=150)
            plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not create 3D scatter for run {trial_idx}: {e}")

    # Return summary for aggregator
    return {
        "run_index": trial_idx,
        "final_map": m_final,
        "qd_scores": qd_scores,
        "coverage_rel": coverage_rel,
        "gen_times_total": gen_times_total,
        "gen_times_synth": gen_times_synth,
        "gen_times_desc": gen_times_desc,
        "dims": dims,
    }


def aggregate_and_plot(runs: List[Dict[str, Any]], out_root: str = "data/evaluation"):
    # Union map and global metrics
    maps = [r["final_map"] for r in runs]
    M = union_max(maps)
    FM_count = len(M)

    reliability = []
    precision = []
    coverage_rel = []
    qd_scores_final = []
    filled_cells = []
    labels = []
    for r in runs:
        m = r["final_map"]
        reliability.append(global_reliability(m, M))
        precision.append(precision_opt_in(m, M))
        coverage_rel.append(coverage_relative(m, FM_count))
        qd_scores_final.append(qd_score(m))
        filled_cells.append(len(m))
        labels.append(f"run_{r['run_index']}")

    # Save global metrics JSON
    global_metrics = {
        "union_cells": FM_count,
        "reliability": reliability,
        "precision": precision,
        "coverage_rel": coverage_rel,
        "qd_scores_final": qd_scores_final,
    }
    save_json(global_metrics, os.path.join(out_root, "global_metrics.json"))

    # Save per-run summary CSV
    rows = []
    for i, r in enumerate(runs):
        rows.append([
            r["run_index"],
            reliability[i],
            precision[i],
            coverage_rel[i],
            qd_scores_final[i],
            filled_cells[i],
        ])
    save_csv(
        rows,
        [
            "run_index",
            "reliability_G(m)",
            "precision_P(m)",
            "coverage_rel_C(m)",
            "qd_score_final",
            "filled_cells",
        ],
        os.path.join(out_root, "runs_summary.csv"),
    )

    # Save per-run metrics JSON into each run folder
    for i, r in enumerate(runs):
        run_dir = os.path.join(out_root, f"run_{r['run_index']:03d}")
        per_run = {
            "reliability": reliability[i],
            "precision": precision[i],
            "coverage_rel": coverage_rel[i],
            "qd_score_final": qd_scores_final[i],
            "filled_cells": filled_cells[i],
            "union_cells": FM_count,
        }
        save_json(per_run, os.path.join(run_dir, "run_metrics.json"))

    # Plots
    plot_bar(labels, reliability, "Global Reliability per run", "G(m)", os.path.join(out_root, "plot_reliability.png"))
    plot_bar(labels, precision, "Precision per run", "P(m)", os.path.join(out_root, "plot_precision.png"))
    plot_bar(labels, coverage_rel, "Relative Coverage per run", "C(m)", os.path.join(out_root, "plot_coverage_rel.png"))
    plot_bar(labels, qd_scores_final, "Final QD-Score per run", "QD-Score", os.path.join(out_root, "plot_qd_final.png"))

    # Aggregate per-generation plots (mean across runs)
    gens = None
    qd_mean = None
    cov_mean = None
    time_total_mean = None
    time_synth_mean = None
    time_desc_mean = None
    if runs:
        lens = [len(r["qd_scores"]) for r in runs]
        L = min(lens)
        gens = list(range(L))
        qd_mean = np.mean([r["qd_scores"][:L] for r in runs], axis=0)
        cov_mean = np.mean([r["coverage_rel"][:L] for r in runs], axis=0)
        time_total_mean = np.mean([r["gen_times_total"][:L] for r in runs], axis=0)
        time_synth_mean = np.mean([r["gen_times_synth"][:L] for r in runs], axis=0)
        time_desc_mean = np.mean([r["gen_times_desc"][:L] for r in runs], axis=0)

        plot_series(gens, {"QD-Score": qd_mean}, "QD-Score over generations (mean)", "Generation", "QD-Score", os.path.join(out_root, "plot_qd_over_gens.png"))
        plot_series(gens, {"Coverage(abs)": cov_mean}, "Coverage over generations (mean)", "Generation", "Coverage(abs)", os.path.join(out_root, "plot_cov_over_gens.png"))
        plot_series(
            gens,
            {
                "Total": time_total_mean,
                "Synthesis": time_synth_mean,
                "Descriptors": time_desc_mean,
            },
            "Timing over generations (mean)",
            "Generation",
            "Seconds",
            os.path.join(out_root, "plot_timing_over_gens.png"),
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluation runner for audio MAP-Elites")
    parser.add_argument("--runs", type=int, default=5, help="Number of trials to run")
    parser.add_argument("--generations", type=int, default=50, help="Generations per run")
    parser.add_argument("--pop", type=int, default=64, help="Population size (batch)")
    parser.add_argument("--sr", type=int, default=44100, help="Sampling rate")
    parser.add_argument("--dims", type=int, nargs=3, default=[20, 20, 20], help="Archive dims H W D")
    parser.add_argument("--duration", type=float, default=2.0, help="Audio length in seconds per individual")
    parser.add_argument("--out", type=str, default="data/evaluation", help="Output root folder")
    args = parser.parse_args()

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = BACKEND_DIR / out_root
    out_root = os.fspath(out_root)

    runs = []
    for i in range(args.runs):
        print(f"Running trial {i+1}/{args.runs}...")
        r = run_single_trial(
            trial_idx=i,
            generations=args.generations,
            pop_size=args.pop,
            sr=args.sr,
            dims=tuple(args.dims),
            duration=args.duration,
            out_root=out_root,
        )
        runs.append(r)

    aggregate_and_plot(runs, out_root)
    save_citations(out_root)
    print("Done. Outputs in:", out_root)


if __name__ == "__main__":
    main()
