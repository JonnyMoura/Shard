import numpy as np

# Local registries for test runs
RUN_TRACK_ASSIGNMENTS = {}
RUN_CATEGORY_ASSIGNMENTS = {}

TRACKS_PER_SOLUTION = 2
CATEGORIES = ["low", "mid", "high", "rhythmic"]


def clear_registries():
    """Clear all registries between runs to ensure independent state."""
    global RUN_TRACK_ASSIGNMENTS, RUN_CATEGORY_ASSIGNMENTS
    RUN_TRACK_ASSIGNMENTS = {}
    RUN_CATEGORY_ASSIGNMENTS = {}


def solution_to_id(solution: np.ndarray) -> int:
    return hash(tuple(np.asarray(solution).round(6)))


def get_assigned_category(solution: np.ndarray) -> str | None:
    sid = solution_to_id(solution)
    return RUN_CATEGORY_ASSIGNMENTS.get(sid)


def get_tracks_from_solution(solution: np.ndarray, catalog_df) -> list[str]:
    sid = solution_to_id(solution)
    if sid in RUN_TRACK_ASSIGNMENTS:
        idxs = RUN_TRACK_ASSIGNMENTS[sid]
    else:
        # Fallback random tracks
        idxs = [np.random.randint(0, len(catalog_df)) for _ in range(TRACKS_PER_SOLUTION)]
        RUN_TRACK_ASSIGNMENTS[sid] = idxs
    idxs = [int(np.clip(i, 0, len(catalog_df) - 1)) for i in idxs]
    return [catalog_df.iloc[i]["file_path"] for i in idxs]


def balanced_category_track_assignment(offspring: np.ndarray, catalog_df):
    pools = {}
    for cat in CATEGORIES:
        pools[cat] = catalog_df[catalog_df["dominant_category"] == cat].index.tolist()

    n = len(offspring)
    per = n // 4
    rem = n % 4
    plan = []
    for i, cat in enumerate(CATEGORIES):
        c = per + (1 if i < rem else 0)
        plan.extend([cat] * c)
    np.random.shuffle(plan)

    for sol, cat in zip(offspring, plan):
        sid = solution_to_id(sol)
        RUN_CATEGORY_ASSIGNMENTS[sid] = cat
        assigned = []
        if np.random.random() < 0.7:
            # Both tracks from same category pool
            for _ in range(TRACKS_PER_SOLUTION):
                if pools[cat]:
                    assigned.append(int(np.random.choice(pools[cat])))
                else:
                    assigned.append(np.random.randint(0, len(catalog_df)))
        else:
            # Mix: first from cat, second from complementary
            if pools[cat]:
                assigned.append(int(np.random.choice(pools[cat])))
            else:
                assigned.append(np.random.randint(0, len(catalog_df)))
            comp = np.random.choice([c for c in CATEGORIES if c != cat])
            if pools[comp]:
                assigned.append(int(np.random.choice(pools[comp])))
            else:
                assigned.append(np.random.randint(0, len(catalog_df)))
        RUN_TRACK_ASSIGNMENTS[sid] = assigned
    return offspring


def distance_threshold(general_bounds):
    """Calculate a reasonable distance threshold based on parameter space size (like main.py)."""
    param_ranges = general_bounds[1] - general_bounds[0]
    diagonal_length = np.linalg.norm(param_ranges)
    # Use 15% of diagonal as threshold
    return 0.15 * diagonal_length


def find_closest_archive_elite(solution, archive):
    """Find the archive elite closest to this solution and return both elite and distance."""
    min_distance = float('inf')
    closest_elite = None

    if len(archive) == 0:
        return None, float('inf')

    for elite in archive:
        if elite is not None:
            elite_solution = elite.get("solution") if hasattr(elite, "get") else getattr(elite, "solution", None)
            if elite_solution is not None:
                distance = np.linalg.norm(solution - elite_solution)
                if distance < min_distance:
                    min_distance = distance
                    closest_elite = elite_solution

    return closest_elite, min_distance


def assign_inheritance_pairs(offspring, archive, min_distance_threshold):
    """Assign inheritance pairs: each archive elite gets matched to its closest offspring if close enough."""
    if len(archive) == 0:
        return set()

    # Get all archive elites
    archive_elites = []
    for elite in archive:
        if elite is not None:
            elite_solution = elite.get("solution") if hasattr(elite, "get") else getattr(elite, "solution", None)
            if elite_solution is not None:
                archive_elites.append(elite_solution)

    if len(archive_elites) == 0:
        return set()

    inheritance_assignments = set()

    # For each archive elite, find its closest offspring
    for archive_elite in archive_elites:
        min_distance = float('inf')
        closest_offspring_idx = None

        for i, offspring_sol in enumerate(offspring):
            if i in inheritance_assignments:
                continue
            distance = np.linalg.norm(offspring_sol - archive_elite)
            if distance < min_distance:
                min_distance = distance
                closest_offspring_idx = i

        # Only assign inheritance if offspring is close enough
        if closest_offspring_idx is not None and min_distance <= min_distance_threshold:
            inheritance_assignments.add(closest_offspring_idx)

    return inheritance_assignments


def inherit_tracks_from_archive(offspring: np.ndarray, catalog_df, archive, general_bounds, mutation_rate: float = 0.15):
    """Inherit tracks from archive parents - only closest offspring to each elite inherits (if close enough).
    
    This matches main.py's inherit_tracks_from_archive logic.
    """
    pools = {cat: catalog_df[catalog_df["dominant_category"] == cat].index.tolist() for cat in CATEGORIES}

    # Calculate distance threshold
    dist_thresh = distance_threshold(general_bounds)

    # Determine which offspring should inherit (closest to each archive elite, if close enough)
    inheritor_indices = assign_inheritance_pairs(offspring, archive, dist_thresh)

    for i, sol in enumerate(offspring):
        sid = solution_to_id(sol)

        if i in inheritor_indices:
            # Find the closest archive elite as parent
            parent_solution, parent_distance = find_closest_archive_elite(sol, archive)

            if parent_solution is not None and parent_distance <= dist_thresh:
                parent_id = solution_to_id(parent_solution)

                if parent_id in RUN_TRACK_ASSIGNMENTS:
                    # INHERIT from parent
                    parent_tracks = list(RUN_TRACK_ASSIGNMENTS[parent_id])
                    parent_category = RUN_CATEGORY_ASSIGNMENTS.get(parent_id, np.random.choice(CATEGORIES))

                    # Apply track mutations
                    for t in range(len(parent_tracks)):
                        if np.random.random() < mutation_rate:
                            # Mutate track within category or complementary
                            mutation_type = np.random.random()
                            if mutation_type < 0.7:  # Same category
                                if pools[parent_category]:
                                    parent_tracks[t] = int(np.random.choice(pools[parent_category]))
                            elif mutation_type < 0.85:  # Complementary
                                comp_map = {'low': 'high', 'high': 'low', 'mid': 'rhythmic', 'rhythmic': 'mid'}
                                comp_cat = comp_map.get(parent_category, parent_category)
                                if pools[comp_cat]:
                                    parent_tracks[t] = int(np.random.choice(pools[comp_cat]))
                            else:  # Random cross-category
                                other_cats = [c for c in CATEGORIES if c != parent_category]
                                if other_cats:
                                    target_cat = np.random.choice(other_cats)
                                    if pools[target_cat]:
                                        parent_tracks[t] = int(np.random.choice(pools[target_cat]))

                    RUN_TRACK_ASSIGNMENTS[sid] = parent_tracks
                    RUN_CATEGORY_ASSIGNMENTS[sid] = parent_category
                    continue

        # This offspring gets NEW track assignment (not inherited)
        assigned_category = np.random.choice(CATEGORIES)
        assigned_tracks = []

        # 80% same category, 20% complementary mix (like main.py)
        if np.random.random() < 0.8:
            for _ in range(TRACKS_PER_SOLUTION):
                if pools[assigned_category]:
                    assigned_tracks.append(int(np.random.choice(pools[assigned_category])))
                else:
                    assigned_tracks.append(np.random.randint(0, len(catalog_df)))
        else:
            if pools[assigned_category]:
                assigned_tracks.append(int(np.random.choice(pools[assigned_category])))
            else:
                assigned_tracks.append(np.random.randint(0, len(catalog_df)))
            comp_map = {'low': 'mid', 'high': 'mid', 'mid': 'rhythmic', 'rhythmic': 'mid'}
            comp_cat = comp_map[assigned_category]
            if pools[comp_cat]:
                assigned_tracks.append(int(np.random.choice(pools[comp_cat])))
            else:
                assigned_tracks.append(np.random.randint(0, len(catalog_df)))

        RUN_TRACK_ASSIGNMENTS[sid] = assigned_tracks
        RUN_CATEGORY_ASSIGNMENTS[sid] = assigned_category

    return offspring


# Keep old function for backward compatibility but redirect to archive-based
def inherit_tracks_from_previous(offspring: np.ndarray, catalog_df, prev_population: np.ndarray, mutation_rate: float = 0.15, archive=None, general_bounds=None):
    """Legacy wrapper - now redirects to archive-based inheritance if archive provided."""
    if archive is not None and general_bounds is not None:
        return inherit_tracks_from_archive(offspring, catalog_df, archive, general_bounds, mutation_rate)
    
    # Fallback to simple random-parent inheritance if no archive
    pools = {cat: catalog_df[catalog_df["dominant_category"] == cat].index.tolist() for cat in CATEGORIES}
    if prev_population is None:
        prev_sids = []
    else:
        try:
            prev_sids = [solution_to_id(sol) for sol in list(prev_population)]
        except Exception:
            prev_sids = []

    for sol in offspring:
        sid = solution_to_id(sol)
        parent_sid = np.random.choice(prev_sids) if prev_sids else None
        cat = RUN_CATEGORY_ASSIGNMENTS.get(sid)
        if cat is None and parent_sid is not None:
            cat = RUN_CATEGORY_ASSIGNMENTS.get(parent_sid, np.random.choice(CATEGORIES))
            RUN_CATEGORY_ASSIGNMENTS[sid] = cat
        if cat is None:
            cat = np.random.choice(CATEGORIES)
            RUN_CATEGORY_ASSIGNMENTS[sid] = cat
        if parent_sid is not None and parent_sid in RUN_TRACK_ASSIGNMENTS:
            inherited = list(RUN_TRACK_ASSIGNMENTS[parent_sid])
        else:
            inherited = RUN_TRACK_ASSIGNMENTS.get(sid, [np.random.randint(0, len(catalog_df)) for _ in range(TRACKS_PER_SOLUTION)])
        for t in range(TRACKS_PER_SOLUTION):
            if np.random.random() < mutation_rate:
                if pools[cat]:
                    inherited[t] = int(np.random.choice(pools[cat]))
                else:
                    inherited[t] = np.random.randint(0, len(catalog_df))
        RUN_TRACK_ASSIGNMENTS[sid] = inherited
    return offspring
