import concurrent.futures
import multiprocessing as mp
import os

from audio_synthesis import extract_audio_descriptors, generate_audio_stream


def resolve_source_path(path, data_dir, source_dir):
    if os.path.isabs(path):
        return path

    candidates = (
        os.path.join(os.path.dirname(data_dir), path),
        os.path.join(data_dir, path),
        os.path.join(source_dir, path),
    )
    return next(
        (candidate for candidate in candidates if os.path.isfile(candidate)),
        candidates[0],
    )


def synthesize_job(job):
    """Synthesize one solution using source assignments prepared by the caller."""
    source_tracks = job["source_tracks"]
    audio_stream, sample_rate = generate_audio_stream(
        job["solution"],
        duration=job["duration"],
        sources=source_tracks,
        category=job["category"],
    )
    descriptors = extract_audio_descriptors(audio_stream, sample_rate)
    if not job["include_audio"]:
        audio_stream = None
    return audio_stream, sample_rate, descriptors, source_tracks


def run_synthesis_jobs(jobs):
    if not jobs:
        return []

    results = [None] * len(jobs)
    max_workers = min(mp.cpu_count(), len(jobs), 4)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(synthesize_job, job) for job in jobs]
        for index, future in enumerate(futures):
            try:
                results[index] = future.result()
            except Exception as error:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"Synthesis failed for solution {index}") from error
    return results


def render_selected_previews(selected_indices, jobs, results):
    preview_jobs = []
    for index in selected_indices:
        preview_job = dict(jobs[index])
        preview_job["duration"] = 30
        preview_job["include_audio"] = True
        preview_jobs.append(preview_job)

    for index, preview in zip(selected_indices, run_synthesis_jobs(preview_jobs)):
        _, _, descriptors, source_tracks = results[index]
        results[index] = (preview[0], preview[1], descriptors, source_tracks)
