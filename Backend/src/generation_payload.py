import os
import time

import numpy as np


CATEGORIES = ("low", "mid", "high", "rhythmic")


def build_generation_payload(
    generation_num,
    offspring,
    results,
    objective_batch,
    categories_list,
    inheritor_indices,
    get_assigned_category,
    audio_to_base64,
):
    generation_data = {
        "type": "generation_data",
        "generation": generation_num,
        "timestamp": int(time.time() * 1000),
        "solutions": [],
    }

    print(f"🐍 Processing {len(offspring)} solutions...")
    for index, (solution, result) in enumerate(zip(offspring, results)):
        audio_stream, sample_rate, descriptors, source_tracks = result
        assigned_category = get_assigned_category(solution)
        actual_category = categories_list[index]
        fitness = float(objective_batch[index])
        is_inherited = index in inheritor_indices

        print(f"🐍 Converting audio for solution {index + 1}/{len(offspring)}")
        if audio_stream is None:
            raise RuntimeError(f"Selected solution {index} has no preview audio")
        audio_base64 = audio_to_base64(audio_stream, sample_rate)

        solution_data = {
            "id": index,
            "solution_id": f"solution_{generation_num:03d}_{index:03d}",
            "parameters": {
                "grain_duration": float(solution[0]),
                "overlap": float(solution[1]),
                "pitch_variance": float(solution[2]),
                "start_offset_range": float(solution[3]),
                "room_size": float(solution[4]),
                "damping": float(solution[5]),
                "wet_level": float(solution[6]),
                "num_voices": int(solution[7]),
                "low_cutoff_hz": float(solution[8]),
                "high_cutoff_hz": float(solution[9]),
                "ladder_resonance": float(solution[10]),
            },
            "descriptors": {
                "centroid": float(descriptors[0]),
                "rolloff": float(descriptors[1]),
                "spectral_flux": float(descriptors[2]),
                "onset_rate": float(descriptors[3]),
                "zcr": float(descriptors[4]) if len(descriptors) > 4 else 0.0,
            },
            "fitness": fitness,
            "assigned_category": assigned_category,
            "actual_category": actual_category,
            "inherited": is_inherited,
            "sources": [os.path.basename(path) for path in source_tracks],
            "audio": {
                "data": audio_base64,
                "sample_rate": int(sample_rate),
                "duration": len(audio_stream) / sample_rate,
            },
        }

        generation_data["solutions"].append(solution_data)
        print(f"  ✅ Solution {index + 1} data prepared")

    generation_data["statistics"] = {
        "total_solutions": len(offspring),
        "inherited_count": len(inheritor_indices),
        "new_count": len(offspring) - len(inheritor_indices),
        "avg_fitness": float(np.mean(objective_batch)),
        "best_fitness": float(np.max(objective_batch)),
        "worst_fitness": float(np.min(objective_batch)),
        "category_distribution": {
            category: categories_list.count(category) for category in CATEGORIES
        },
    }
    return generation_data
