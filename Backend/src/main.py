import os
import numpy as np
import json
from dataset_class import analyze_folder, classify_audio_category
import soundfile as sf
from evolutionary_module import archive, emitters, automatic_fitness, general_bounds
from ribs.schedulers import Scheduler
import pandas as pd
import asyncio
import websockets
import threading
import base64
import io
import queue
from synthesis_pipeline import (
    render_selected_previews as _render_selected_previews,
    resolve_source_path,
    run_synthesis_jobs as _run_synthesis_jobs,
)
from feedback_validation import feedback_matches_generation as _feedback_matches_generation
from generation_payload import build_generation_payload

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

TRACKS_PER_SOLUTION = 2  # Number of tracks per solution
categories = ['low', 'mid', 'high', 'rhythmic']

# Global dictionaries to store track assignments
SOLUTION_TRACK_ASSIGNMENTS = {}  # Maps solution_id -> [track_indices]
SOLUTION_CATEGORY_ASSIGNMENTS = {}  # Maps solution_id -> assigned_category

# WebSocket server globals
websocket_clients = set()
websocket_loop = None
generation_data_latest = None  # Replace generation_data_queue = []
user_feedback_queue = queue.Queue()  
evolution_state = {
    'running': False,
    'current_generation': 0,
    'waiting_for_feedback': True,  
    'scheduler': None,
    'catalog_df': None,
    'inheritor_indices': set(),
    'initial_solutions_sent': False,
    'pending_generation_data': None  # NEW: Store generation data until feedback arrives
}

def solution_to_id(solution):
    """Convert solution array to a hashable ID."""
    return hash(tuple(solution.round(6))) 

def get_tracks_from_solution(solution, catalog_df):
    """Resolve the parent-process track assignment for a solution."""
    solution_id = solution_to_id(solution)
    if solution_id not in SOLUTION_TRACK_ASSIGNMENTS:
        raise KeyError(f"No track assignment exists for solution {solution_id}")
    if catalog_df.empty:
        raise ValueError("The audio catalog is empty")

    track_indices = SOLUTION_TRACK_ASSIGNMENTS[solution_id]
    return [catalog_df.iloc[int(idx)]["file_path"] for idx in track_indices]

def balanced_category_track_assignment(offspring, catalog_df):
    """Assign tracks to solutions with balanced category distribution using global dicts."""
    
    # Create category pools
    category_pools = {}
    for category in ['low', 'mid', 'high', 'rhythmic']:
        category_tracks = catalog_df[catalog_df['dominant_category'] == category].index.tolist()
        category_pools[category] = category_tracks
        print(f"Category '{category}': {len(category_tracks)} tracks available")
    
    categories = ['low', 'mid', 'high', 'rhythmic']
    num_solutions = len(offspring)
    solutions_per_category = num_solutions // 4
    remaining_solutions = num_solutions % 4
    
    # Create balanced assignment plan
    assignment_plan = []
    for i, category in enumerate(categories):
        count = solutions_per_category
        if i < remaining_solutions:  # Distribute remaining solutions
            count += 1
        assignment_plan.extend([category] * count)
    
    # Shuffle to avoid systematic bias
    np.random.shuffle(assignment_plan)
    
    print(f"Category assignment plan: {dict(zip(*np.unique(assignment_plan, return_counts=True)))}")
    
    # Assign tracks based on the plan
    for j, (sol, assigned_category) in enumerate(zip(offspring, assignment_plan)):
        solution_id = solution_to_id(sol)
        
        # Both tracks from same category (70% of time)
        if np.random.random() < 0.7:
            assigned_tracks = []
            for track_slot in range(TRACKS_PER_SOLUTION):
                if category_pools[assigned_category]:
                    track_idx = np.random.choice(category_pools[assigned_category])
                    assigned_tracks.append(track_idx)
                else:
                    # Fallback to random track
                    assigned_tracks.append(np.random.randint(0, len(catalog_df)))
        
        # Mix with complementary category (30% of time)
        else:
            assigned_tracks = []
            
            # First track from assigned category
            if category_pools[assigned_category]:
                track_idx = np.random.choice(category_pools[assigned_category])
                assigned_tracks.append(track_idx)
            else:
                assigned_tracks.append(np.random.randint(0, len(catalog_df)))
            
            # Second track from complementary category
            complementary_map = {
                'low': 'mid',
                'high': 'mid', 
                'mid': 'rhythmic',
                'rhythmic': 'mid'
            }
            complementary_category = complementary_map[assigned_category]
            
            if category_pools[complementary_category]:
                track_idx = np.random.choice(category_pools[complementary_category])
                assigned_tracks.append(track_idx)
            else:
                assigned_tracks.append(np.random.randint(0, len(catalog_df)))
        
        # Store assignments in global dictionaries
        SOLUTION_TRACK_ASSIGNMENTS[solution_id] = assigned_tracks
        SOLUTION_CATEGORY_ASSIGNMENTS[solution_id] = assigned_category
    
    return offspring

def get_assigned_category(solution):
    """Get assigned category for a solution."""
    solution_id = solution_to_id(solution)
    return SOLUTION_CATEGORY_ASSIGNMENTS.get(solution_id, 'unknown')

def _make_synthesis_jobs(offspring, catalog_df, duration, include_audio):
    jobs = []
    for solution in offspring:
        source_tracks = [
            resolve_source_path(path, DATA_DIR, os.path.dirname(__file__))
            for path in get_tracks_from_solution(solution, catalog_df)
        ]
        jobs.append({
            "solution": solution,
            "source_tracks": source_tracks,
            "category": get_assigned_category(solution),
            "duration": duration,
            "include_audio": include_audio,
        })
    return jobs


def assign_category_from_descriptors(descriptors):
    centroid, _, spectral_flux, onset_rate, zcr = descriptors
    return classify_audio_category(centroid, spectral_flux, onset_rate, zcr)

def find_closest_archive_elite(solution, archive):
    """Find the archive elite closest to this solution and return both elite and distance."""
    min_distance = float('inf')
    closest_elite = None
    
    # Check if archive is empty
    if len(archive) == 0:
        return None, float('inf')
    
    # Iterate through all elites in the archive
    for elite in archive:
        if elite is not None:
            elite_solution = elite.get("solution") 
            distance = np.linalg.norm(solution - elite_solution)
            if distance < min_distance:
                min_distance = distance
                closest_elite = elite_solution
    
    return closest_elite, min_distance

def distance_threshold(general_bounds):
    """Calculate a reasonable distance threshold based on parameter space size."""
    # Get the range of each parameter
    param_ranges = general_bounds[1] - general_bounds[0]
    
    # Calculate diagonal of parameter space
    diagonal_length = np.linalg.norm(param_ranges)

    # Use 15% of diagonal as threshold 
    threshold = 0.15 * diagonal_length
    
    return threshold

def assign_inheritance_pairs(offspring, archive, min_distance_threshold):
    """Assign inheritance pairs: each archive elite gets matched to its closest offspring if close enough."""
    if len(archive) == 0:
        return set()  # No inheritance possible
    
    # Get all archive elites
    archive_elites = []
    for elite in archive:
        if elite is not None:
            archive_elites.append(elite.get("solution"))  

    if len(archive_elites) == 0:
        return set()
    
    inheritance_assignments = set()
    rejected_distances = []
    
    # For each archive elite, find its closest offspring 
    for archive_elite in archive_elites:
        min_distance = float('inf')
        closest_offspring_idx = None
        
        for i, offspring_sol in enumerate(offspring):
            # Skip if this offspring was already assigned to inherit
            if i in inheritance_assignments:
                continue
                
            distance = np.linalg.norm(offspring_sol - archive_elite)
            if distance < min_distance:
                min_distance = distance
                closest_offspring_idx = i
        
        # Only assign inheritance if offspring is close enough
        if closest_offspring_idx is not None and min_distance <= min_distance_threshold:
            inheritance_assignments.add(closest_offspring_idx)
        elif min_distance > min_distance_threshold:
            rejected_distances.append(min_distance)
    
    if rejected_distances:
        print(f"  Rejected {len(rejected_distances)} inheritance pairs (distances too large: min={min(rejected_distances):.3f}, max={max(rejected_distances):.3f}, threshold={min_distance_threshold:.3f})")
    
    return inheritance_assignments

def inherit_tracks_from_archive(offspring, catalog_df, archive, mutation_rate):
    """Inherit tracks from archive parents - only closest offspring to each elite inherits (if close enough)."""
    
    category_pools = {}
    for category in ['low', 'mid', 'high', 'rhythmic']:
        category_tracks = catalog_df[catalog_df['dominant_category'] == category].index.tolist()
        category_pools[category] = category_tracks
    
    new_solutions_count = 0
    inherited_solutions_count = 0
    mutated_tracks_count = 0
    
    # Calculate reasonable distance threshold based on parameter space
    distance_threshold_aux = distance_threshold(general_bounds)
    print(f"  Inheritance distance threshold: {distance_threshold_aux:.3f}")
    
    # Determine which offspring should inherit (closest to each archive elite, if close enough)
    inheritor_indices = assign_inheritance_pairs(offspring, archive, distance_threshold_aux)
    
    print(f"Inheritance plan: {len(inheritor_indices)} offspring will inherit, {len(offspring) - len(inheritor_indices)} will get new assignments")
    
    for i, sol in enumerate(offspring):
        solution_id = solution_to_id(sol)
        
        #
        if i in inheritor_indices:
            # Find the closest archive elite as parent
            parent_solution, parent_distance = find_closest_archive_elite(sol, archive)
            
            if parent_solution is not None and parent_distance <= distance_threshold_aux:
                parent_id = solution_to_id(parent_solution)
                
                if parent_id in SOLUTION_TRACK_ASSIGNMENTS:
                    # INHERIT from parent
                    inherited_solutions_count += 1
                    parent_tracks = SOLUTION_TRACK_ASSIGNMENTS[parent_id].copy()
                    parent_category = SOLUTION_CATEGORY_ASSIGNMENTS[parent_id]
                    
                    print(f"  Offspring {i+1} inheriting from parent (distance: {parent_distance:.3f}, category: {parent_category})")
                    
                    # Apply track mutations
                    for track_idx in range(len(parent_tracks)):
                        if np.random.random() < mutation_rate:
                            mutated_tracks_count += 1
                            current_track = parent_tracks[track_idx]
                            
                            if 0 <= current_track < len(catalog_df):
                                track_category = catalog_df.iloc[current_track]['dominant_category']
                            else:
                                track_category = parent_category
                            
                            
                            mutation_type = np.random.random()
                            if mutation_type < 0.7:  # 50% to 70% - Same category
                                if category_pools[track_category]:
                                    parent_tracks[track_idx] = np.random.choice(category_pools[track_category])
                            elif mutation_type < 0.80:  # 30% to 15% - Complementary
                                complementary_map = {
                                    'low': 'high', 'high': 'low', 
                                    'mid': 'rhythmic', 'rhythmic': 'mid'
                                }
                                comp_category = complementary_map.get(track_category, track_category)
                                if category_pools[comp_category]:
                                    parent_tracks[track_idx] = np.random.choice(category_pools[comp_category])
                            else:  #  30% to 15% - Random cross-category
                                other_categories = [cat for cat in categories if cat != track_category]
                                if other_categories:
                                    target_category = np.random.choice(other_categories)
                                    if category_pools[target_category]:
                                        parent_tracks[track_idx] = np.random.choice(category_pools[target_category])
                    
                    # Store inherited tracks for child
                    SOLUTION_TRACK_ASSIGNMENTS[solution_id] = parent_tracks
                    SOLUTION_CATEGORY_ASSIGNMENTS[solution_id] = parent_category
                    continue
        
        # This offspring gets NEW track assignment 
        new_solutions_count += 1
        
        # Use balanced assignment for new solutions
        assigned_category = np.random.choice(categories)
        assigned_tracks = []
        
        print(f"  Offspring {i+1} getting new assignment (category: {assigned_category})")

        # 80% same category, 20% complementary mix
        if np.random.random() < 0.8:
            # Both tracks from same category
            for _ in range(TRACKS_PER_SOLUTION):
                if category_pools[assigned_category]:
                    track_idx = np.random.choice(category_pools[assigned_category])
                    assigned_tracks.append(track_idx)
                else:
                    assigned_tracks.append(np.random.randint(0, len(catalog_df)))
        else:
            # Mix with complementary category
            # First track from assigned category
            if category_pools[assigned_category]:
                track_idx = np.random.choice(category_pools[assigned_category])
                assigned_tracks.append(track_idx)
            else:
                assigned_tracks.append(np.random.randint(0, len(catalog_df)))
            
            # Second track from complementary category
            complementary_map = {
                'low': 'mid',
                'high': 'mid', 
                'mid': 'rhythmic',
                'rhythmic': 'mid'
            }
            complementary_category = complementary_map[assigned_category]
            
            if category_pools[complementary_category]:
                track_idx = np.random.choice(category_pools[complementary_category])
                assigned_tracks.append(track_idx)
            else:
                assigned_tracks.append(np.random.randint(0, len(catalog_df)))
        
        SOLUTION_TRACK_ASSIGNMENTS[solution_id] = assigned_tracks
        SOLUTION_CATEGORY_ASSIGNMENTS[solution_id] = assigned_category
    
    print(f"Track inheritance summary: {inherited_solutions_count} inherited, {new_solutions_count} new")
    print(f"Track mutations: {mutated_tracks_count} tracks mutated")
    
    return offspring

def calculate_hybrid_fitness(descriptors, category, user_score=None, was_scored=False):
    """Calculate hybrid fitness combining automatic fitness with user feedback."""
    auto_fitness = automatic_fitness(descriptors, category)
    
    if was_scored and user_score is not None:
        # User feedback has higher weight (70% user, 30% automatic) - ONLY for user scored solutions
        hybrid_fitness = 0.7 * user_score + 0.3 * auto_fitness
        print(f"    👤 Hybrid fitness: user={user_score:.3f} + auto={auto_fitness:.3f} = {hybrid_fitness:.3f}")
        return hybrid_fitness
    else:
        # Use pure automatic fitness for unscored solutions
        print(f"    🤖 Auto fitness: {auto_fitness:.3f}")
        return auto_fitness

def audio_to_base64(audio_stream, sample_rate):
    """Convert audio stream to base64 for transmission, with normalization."""
    try:
        buffer = io.BytesIO()

        # Ensure audio is in correct format
        if audio_stream.ndim > 1:
            audio_stream = audio_stream.mean(axis=1)  # Convert to mono

        
        max_val = np.max(np.abs(audio_stream))
        if max_val > 0:
            audio_normalized = audio_stream / max_val * 0.99
        else:
            audio_normalized = audio_stream

        audio_normalized = np.clip(audio_normalized, -1.0, 1.0)

        # Write to WAV format
        sf.write(buffer, audio_normalized, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)

        # Convert to base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        print(f"    Audio conversion: {len(audio_normalized)} samples → {len(audio_base64)} base64 chars")
        return audio_base64

    except Exception as e:
        print(f"    ❌ Audio conversion error: {e}")
        raise RuntimeError("Could not encode synthesized audio") from e

async def _broadcast_generation(generation_data):
    message = json.dumps(generation_data)
    for client in tuple(websocket_clients):
        try:
            await client.send(message)
        except websockets.exceptions.ConnectionClosed:
            websocket_clients.discard(client)
        except Exception as error:
            print(f"WebSocket broadcast failed for {client.remote_address}: {error}")
            websocket_clients.discard(client)


def start_websocket_server():
    """Run the WebSocket server on its own event loop."""
    def run_server():
        global websocket_loop

        async def websocket_handler(websocket):
            print(f"New WebSocket client connected from {websocket.remote_address}")
            websocket_clients.add(websocket)
            try:
                if generation_data_latest is not None:
                    await websocket.send(json.dumps(generation_data_latest))

                async for message in websocket:
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        print(f"Invalid JSON received: {message}")
                        continue

                    if not isinstance(data, dict):
                        print(f"Invalid WebSocket message object: {data}")
                        continue

                    if (
                        data.get("action") == "user_feedback"
                        and data.get("type") == "user_scoring"
                    ):
                        pending = evolution_state.get("pending_generation_data") or {}
                        expected_count = len(pending.get("selected_indices", []))
                        if _feedback_matches_generation(
                            data,
                            evolution_state["current_generation"],
                            expected_count,
                        ):
                            user_feedback_queue.put(data)
                        else:
                            print(
                                "Discarding stale or malformed feedback for generation "
                                f"{data.get('generation')}; expected "
                                f"{evolution_state['current_generation']}."
                            )
                    else:
                        print(f"Unknown message type: {data}")
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket client disconnected")
            except Exception as error:
                print(f"WebSocket error: {error}")
            finally:
                websocket_clients.discard(websocket)

        async def serve_forever():
            server = await websockets.serve(websocket_handler, "localhost", 8765)
            print("WebSocket server started on ws://localhost:8765")
            async with server:
                await server.wait_closed()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        websocket_loop = loop
        try:
            loop.run_until_complete(serve_forever())
        except KeyboardInterrupt:
            print("WebSocket server stopped")
        finally:
            websocket_loop = None
            loop.close()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    return server_thread

def send_generation_data(generation_num, offspring, results, objective_batch, categories_list, inheritor_indices):
    """Prepare generation data and queue it for WebSocket transmission."""
    global generation_data_latest  # Access the global variable

    print(f"🐍 PREPARING GENERATION {generation_num} DATA FOR WEBSOCKET...")

    generation_data = build_generation_payload(
        generation_num,
        offspring,
        results,
        objective_batch,
        categories_list,
        inheritor_indices,
        get_assigned_category,
        audio_to_base64,
    )

    generation_data_latest = generation_data
    print(f"🐍 GENERATION {generation_num} DATA STORED AS LATEST")
    print(f"  - Solutions: {len(generation_data['solutions'])}")
    loop = websocket_loop
    if loop is not None and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            _broadcast_generation(generation_data), loop
        )

        def report_broadcast_error(completed):
            try:
                completed.result()
            except Exception as error:
                print(f"WebSocket broadcast failed: {error}")

        future.add_done_callback(report_broadcast_error)
    
    return generation_data

def prepare_and_send_initial_solutions(catalog_df):
    """Generate and send the first generation."""
    print("\n📤 GENERATING AND SENDING INITIAL SOLUTIONS...")

    scheduler = evolution_state['scheduler']

    # Generate first batch
    offspring = scheduler.ask()
    for j in range(len(offspring)):
        offspring[j] = np.clip(offspring[j], general_bounds[0], general_bounds[1])

    # Assign tracks
    offspring = balanced_category_track_assignment(offspring, catalog_df)
    inheritor_indices = set()

    synthesis_jobs = _make_synthesis_jobs(
        offspring, catalog_df, duration=2, include_audio=False
    )
    results = _run_synthesis_jobs(synthesis_jobs)

    # Calculate automatic fitness
    objective_batch = []
    categories_list = []

    for j, (sol, result) in enumerate(zip(offspring, results)):
        descriptors = result[2]
        actual_category = assign_category_from_descriptors(descriptors)
        fitness = automatic_fitness(descriptors, actual_category)
        objective_batch.append(fitness)
        categories_list.append(actual_category)

    objective_batch = np.array(objective_batch)

    # Select best 8
    selected_indices = select_best_solutions(
        offspring, objective_batch, categories_list, n_select=8
    )
    _render_selected_previews(selected_indices, synthesis_jobs, results)

    # Store pending data for evaluation when feedback arrives
    # Include selected_indices so we can map user feedback back to original solutions
    evolution_state['pending_generation_data'] = {
        'offspring': offspring,
        'results': results,
        'categories_list': categories_list,
        'inheritor_indices': inheritor_indices,
        'selected_indices': selected_indices  # Maps sent position (0-7) -> original index
    }

    offspring_selected = [offspring[i] for i in selected_indices]
    results_selected = [results[i] for i in selected_indices]
    objective_batch_selected = [objective_batch[i] for i in selected_indices]
    categories_list_selected = [categories_list[i] for i in selected_indices]
    inheritor_indices_selected = set()

    send_generation_data(
        generation_num=0,
        offspring=offspring_selected,
        results=results_selected,
        objective_batch=objective_batch_selected,
        categories_list=categories_list_selected,
        inheritor_indices=inheritor_indices_selected
    )

    evolution_state['initial_solutions_sent'] = True
    evolution_state['current_generation'] = 0
    print("✅ INITIAL SOLUTIONS SENT TO JAVASCRIPT")
    return True
def save_best_solutions(offspring, results, objective_batch, generation_num, save_dir=DATA_DIR):
    """Save the best solutions of the generation to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    best_idx = np.argmax(objective_batch)
    best_solution = offspring[best_idx]
    best_result = results[best_idx]
    best_fitness = float(objective_batch[best_idx])

    # Prepare a dict for saving
    save_dict = {
        "generation": generation_num,
        "fitness": best_fitness,
        "parameters": best_solution.tolist(),
        "descriptors": [float(x) for x in best_result[2]],
        "sources": best_result[3],
        "sample_rate": int(best_result[1])
    }
    save_path = os.path.join(save_dir, f"best_solution_gen_{generation_num:03d}.json")
    with open(save_path, "w") as f:
        json.dump(save_dict, f, indent=2)
    print(f"💾 Best solution of generation {generation_num} saved to {save_path}")


def run_evolution_cycle(user_feedback=None):
    """Run one evolution cycle - generate ONE generation and wait for feedback."""
    
    scheduler = evolution_state['scheduler']
    catalog_df = evolution_state['catalog_df']

    if user_feedback is not None:
        pending = evolution_state.get('pending_generation_data') or {}
        expected_count = len(pending.get('selected_indices', []))
        if not _feedback_matches_generation(
            user_feedback,
            evolution_state['current_generation'],
            expected_count,
        ):
            print(
                "Discarding stale or malformed feedback before evolution: "
                f"received generation {user_feedback.get('generation') if isinstance(user_feedback, dict) else None}, "
                f"expected {evolution_state['current_generation']}."
            )
            return
    
    # STEP 1: Process user feedback for PREVIOUS generation (if exists)
    if user_feedback and evolution_state['pending_generation_data'] is not None:
        print(f"\n📊 PROCESSING USER FEEDBACK FOR GENERATION {evolution_state['current_generation']}")
        
        pending = evolution_state['pending_generation_data']
        offspring = pending['offspring']
        results = pending['results']
        categories_list = pending['categories_list']
        inheritor_indices = pending['inheritor_indices']
        selected_indices = pending.get('selected_indices', [])  # Get the mapping
        
        # Build user feedback lookup - map from SENT position to feedback
        # User sees solutions with IDs solution_XXX_000 through solution_XXX_007
        # These correspond to selected_indices[0], selected_indices[1], etc.
        user_feedback_lookup = {}  # Maps original_index -> feedback_data
        actually_scored_count = 0
        
        if 'feedback' in user_feedback:
            for feedback_item in user_feedback['feedback']:
                solution_id = feedback_item.get('solution_id')  # e.g., "solution_002_003"
                was_scored = feedback_item.get('was_scored', False)
                user_score = feedback_item.get('user_score', 0.0)
                
                if solution_id:
                    try:
                        parts = solution_id.split('_')
                        if len(parts) != 3 or parts[0] != 'solution':
                            raise ValueError(f"Unexpected solution ID format: {solution_id}")
                        solution_generation = int(parts[1])
                        sent_position = int(parts[2])
                        if solution_generation != evolution_state['current_generation']:
                            print(f"  ⚠️ Ignoring feedback from generation {solution_generation}")
                            continue
                        # Map back to original index using selected_indices
                        if sent_position < len(selected_indices):
                            original_index = selected_indices[sent_position]
                            user_feedback_lookup[original_index] = {
                                'user_score': user_score,
                                'was_scored': was_scored,
                                'category': feedback_item.get('category', 'unknown')
                            }
                            
                            if was_scored:
                                actually_scored_count += 1
                                print(f"  📍 Mapped {solution_id} (sent pos {sent_position}) -> original index {original_index}")
                    except (ValueError, IndexError) as e:
                        print(f"  ⚠️ Could not parse solution_id {solution_id}: {e}")
            
            print(f"🎯 Processing user feedback:")
            print(f"  - Total solutions in feedback: {len(user_feedback['feedback'])}")
            print(f"  - Actually scored by user: {actually_scored_count}")
            print(f"  - Selected indices mapping: {selected_indices}")
            
            # Log scored solutions
            scored_solutions = [item for item in user_feedback['feedback'] if item.get('was_scored', False)]
            if scored_solutions:
                print(f"  📊 Actually scored solutions:")
                for item in scored_solutions:
                    print(f"    - {item['solution_id']} ({item['category']}): {item['user_score']:.3f}")
        
        # STEP 2: Calculate hybrid fitness ONLY for the 8 selected solutions
        objective_batch = []
        user_scored_count = 0
        auto_fitness_count = 0
        
        print(f"\n--- Evaluating Generation {evolution_state['current_generation']} (8 selected solutions) ---")
        
        for sent_pos, original_idx in enumerate(selected_indices):
            sol = offspring[original_idx]
            audio_stream, sample_rate, descriptors, source_tracks = results[original_idx]
            actual_category = categories_list[original_idx]
            
            # Look up feedback using ORIGINAL index
            feedback_data = user_feedback_lookup.get(original_idx, {})
            
            user_score = feedback_data.get('user_score', None)
            was_scored = feedback_data.get('was_scored', False)
            
            # Calculate hybrid fitness
            fitness = calculate_hybrid_fitness(
                descriptors, 
                actual_category, 
                user_score=user_score, 
                was_scored=was_scored
            )
            
            objective_batch.append(fitness)
            
            if was_scored:
                user_scored_count += 1
            else:
                auto_fitness_count += 1
            
            feedback_marker = "👤" if was_scored else "🤖"
            print(f"Solution {sent_pos+1} (orig idx {original_idx}) ({feedback_marker}): fitness={fitness:.3f}")
        
        objective_batch = np.array(objective_batch)
        
        print(f"\n📊 Fitness Summary for Generation {evolution_state['current_generation']}:")
        print(f"  - Solutions with user feedback: {user_scored_count}")
        print(f"  - Solutions with auto fitness (selected only): {auto_fitness_count}")
        print(f"  - Total solutions sent to user: {len(selected_indices)}")
        print(f"  - Fitness range: {np.min(objective_batch):.3f} - {np.max(objective_batch):.3f}")
        print(f"  - Average fitness: {np.mean(objective_batch):.3f}")
        
        # STEP 3: Tell archive - ALL 64 solutions, but non-selected have SCALED fitness
        # User-scored solutions can reach ~0.7-1.0, non-selected scaled to max of AUTO_FITNESS_CAP
        # This ensures user feedback always has more value, but non-selected preserve relative ranking
        AUTO_FITNESS_CAP = 0.3  # Maximum fitness for non-selected solutions
        
        full_objective_batch = np.zeros(len(offspring))
        full_measures_batch = []
        selected_set = set(selected_indices)  # For faster lookup
        
        # First pass: calculate auto fitness for ALL non-selected solutions
        non_selected_auto_fitness = {}  # Maps index -> auto_fitness
        for j, (sol, (_, _, descriptors, _)) in enumerate(zip(offspring, results)):
            full_measures_batch.append([descriptors[0], descriptors[2], descriptors[1]])
            
            if j not in selected_set:
                actual_category = categories_list[j]
                auto_fitness = automatic_fitness(descriptors, actual_category)
                non_selected_auto_fitness[j] = auto_fitness
        
        full_measures_batch = np.array(full_measures_batch)
        
        # Second pass: scale non-selected fitness proportionally so max = AUTO_FITNESS_CAP
        # This preserves relative ranking while ensuring none exceed the cap
        if non_selected_auto_fitness:
            max_non_selected = max(non_selected_auto_fitness.values())
            
            if max_non_selected > 0:
                # Scale factor to bring max down to AUTO_FITNESS_CAP
                scale_factor = AUTO_FITNESS_CAP / max_non_selected
                
                for j, auto_fitness in non_selected_auto_fitness.items():
                    # Scale proportionally - best non-selected gets AUTO_FITNESS_CAP, others scaled accordingly
                    scaled_fitness = auto_fitness * scale_factor
                    full_objective_batch[j] = scaled_fitness
            else:
                # All zeros, just set to zero
                for j in non_selected_auto_fitness:
                    full_objective_batch[j] = 0.0
        
        # Set real fitness for the 8 selected solutions (uncapped, includes user feedback)
        for sent_pos, original_idx in enumerate(selected_indices):
            full_objective_batch[original_idx] = objective_batch[sent_pos]
        
        scheduler.tell(full_objective_batch, full_measures_batch)
        
        # Log fitness distribution
        selected_fitness = [full_objective_batch[i] for i in selected_indices]
        non_selected_fitness = [full_objective_batch[i] for i in range(len(offspring)) if i not in selected_set]
        print(f"✅ Generation {evolution_state['current_generation']}:")
        print(f"  - 8 selected: fitness range {min(selected_fitness):.3f} - {max(selected_fitness):.3f}")
        print(f"  - 56 non-selected: scaled to max {AUTO_FITNESS_CAP}, actual range {min(non_selected_fitness):.3f} - {max(non_selected_fitness):.3f}")
        print(f"  Archive coverage: {len(archive)} cells filled")
        
        # Save best solution (from the 8 selected)
        offspring_selected = [offspring[i] for i in selected_indices]
        results_selected = [results[i] for i in selected_indices]
        save_best_solutions(offspring_selected, results_selected, objective_batch, evolution_state['current_generation'], save_dir=DATA_DIR)
        
        # Clear pending data
        evolution_state['pending_generation_data'] = None
    
    # STEP 4: Generate NEXT generation (inherits from updated archive)
    generation_num = evolution_state['current_generation'] + 1
    print(f"\n🧬 GENERATING GENERATION {generation_num}")
    
    offspring = scheduler.ask()
    
    # Clip bounds
    for j in range(len(offspring)):
        offspring[j] = np.clip(offspring[j], general_bounds[0], general_bounds[1])
    
    # Inheritance (uses updated archive with hybrid fitness)
    if generation_num > 0:  # Skip for initial generation
        mutation_rate = 0.15
        print(f"Using archive inheritance (mutation rate: {mutation_rate:.2f})")
        
        distance_threshold_value = distance_threshold(general_bounds)
        inheritor_indices = assign_inheritance_pairs(offspring, archive, distance_threshold_value)
        
        offspring = inherit_tracks_from_archive(offspring, catalog_df, archive, mutation_rate)
    else:
        inheritor_indices = set()
        offspring = balanced_category_track_assignment(offspring, catalog_df)
    
    # Report assignments
    category_assignments = {}
    for sol in offspring:
        cat = get_assigned_category(sol)
        category_assignments[cat] = category_assignments.get(cat, 0) + 1
    print(f"Category distribution: {category_assignments}")
    
    print("Starting synthesis (multiprocessing)")
    synthesis_jobs = _make_synthesis_jobs(
        offspring, catalog_df, duration=2, include_audio=False
    )
    results = _run_synthesis_jobs(synthesis_jobs)
    
    # Calculate AUTOMATIC fitness only (no user feedback yet for THIS generation)
    objective_batch = []
    categories_list = []
    
    print(f"\n--- Generation {generation_num} Solutions (awaiting feedback) ---")
    
    for j, (sol, result) in enumerate(zip(offspring, results)):
        descriptors = result[2]
        actual_category = assign_category_from_descriptors(descriptors)
        fitness = automatic_fitness(descriptors, actual_category)
        
        objective_batch.append(fitness)
        categories_list.append(actual_category)
        
        print(f"Solution {j+1}: category={actual_category}, auto_fitness={fitness:.3f}")
    
    objective_batch = np.array(objective_batch)
    
# STEP 5: Select best 8 first (needed for storing mapping)
    selected_indices = select_best_solutions(
        offspring, objective_batch, categories_list, n_select=8
    )
    _render_selected_previews(selected_indices, synthesis_jobs, results)

    # Store generation data for later evaluation (when feedback arrives)
    # Include selected_indices so we can map user feedback back to original solutions
    evolution_state['pending_generation_data'] = {
        'offspring': offspring,
        'results': results,
        'categories_list': categories_list,
        'inheritor_indices': inheritor_indices,
        'selected_indices': selected_indices  # Maps sent position (0-7) -> original index
    }

    # STEP 6: Send best 8 to user
    
    offspring_selected = [offspring[i] for i in selected_indices]
    results_selected = [results[i] for i in selected_indices]
    objective_batch_selected = [objective_batch[i] for i in selected_indices]
    categories_list_selected = [categories_list[i] for i in selected_indices]
    inheritor_indices_selected = set(j for j, idx in enumerate(selected_indices) if idx in inheritor_indices)
    
    send_generation_data(
        generation_num=generation_num,
        offspring=offspring_selected,
        results=results_selected,
        objective_batch=objective_batch_selected,
        categories_list=categories_list_selected,
        inheritor_indices=inheritor_indices_selected
    )
    
    # Update state
    evolution_state['current_generation'] = generation_num
    evolution_state['waiting_for_feedback'] = True
    
    print(f"\n🎯 Generation {generation_num} sent to user - Waiting for feedback...")

def select_best_solutions(offspring, objective_batch, categories_list, n_select=8):
    """Select n_select best solutions, evenly distributed among categories."""
    categories = ['low', 'mid', 'high', 'rhythmic']
    per_cat = n_select // len(categories)
    selected_indices = []

    # 1. Select top per category
    for cat in categories:
        # Get indices for this category
        cat_indices = [i for i, c in enumerate(categories_list) if c == cat]
        # Sort those indices by fitness
        cat_indices_sorted = sorted(cat_indices, key=lambda i: objective_batch[i], reverse=True)
        
        selected_indices.extend(cat_indices_sorted[:per_cat])

    
    if len(selected_indices) < n_select:
        remaining = n_select - len(selected_indices)
        # Indices not already selected
        not_selected = [i for i in range(len(offspring)) if i not in selected_indices]
        # Sort by fitness
        not_selected_sorted = sorted(not_selected, key=lambda i: objective_batch[i], reverse=True)
        selected_indices.extend(not_selected_sorted[:remaining])

    # 3. Return selected
    return selected_indices



def main():
    print("🚀 Starting Interactive Evolutionary Audio System...")
    
    # Start WebSocket server
    start_websocket_server()
    
    # Load catalog
    try:
        catalog_path = os.path.join(DATA_DIR, "audio_analysis.csv")
        catalog_df = pd.read_csv(catalog_path)
        print(f"Loaded audio catalog with {len(catalog_df)} tracks")
        print(f"Category distribution: {catalog_df['dominant_category'].value_counts().to_dict()}")
        
        scheduler = Scheduler(archive, emitters)
        evolution_state['scheduler'] = scheduler
        evolution_state['catalog_df'] = catalog_df
        
    except Exception as e:
        print(f"Error loading audio catalog: {e}")
        print("Running analyze_folder to create catalog...")
        catalog_df = analyze_folder(
            os.path.join(DATA_DIR, "audio"),
            os.path.join(DATA_DIR, "audio_analysis.csv"),
        )
        
        scheduler = Scheduler(archive, emitters)
        evolution_state['scheduler'] = scheduler
        evolution_state['catalog_df'] = catalog_df
    
    # Create data directory
    data_dir = DATA_DIR
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # Prepare and send initial solutions immediately
    print("\n📤 SENDING STORED SOLUTIONS TO JAVASCRIPT...")
    success = prepare_and_send_initial_solutions(catalog_df)
    
    if success:
        print("\n" + "="*50)
        print("✅ INITIAL SOLUTIONS SENT TO JAVASCRIPT")
        print("🎯 WAITING FOR USER FEEDBACK TO START EVOLUTION")
        print("WebSocket server running on ws://localhost:8765")
        print("Evolution will begin after receiving user feedback")
        print("Press Ctrl+C to exit")
        print("="*50)
    else:
        print("\n❌ FAILED TO SEND INITIAL SOLUTIONS")
        print("Exiting...")
        return
    
    try:
        # Main feedback loop - ONLY start evolution after receiving user feedback
        while True:
            try:
                print(f"⏳ Waiting for user feedback... ({len(websocket_clients)} clients connected)")
                user_feedback = user_feedback_queue.get(timeout=10.0)

                pending = evolution_state.get('pending_generation_data') or {}
                expected_count = len(pending.get('selected_indices', []))
                if not _feedback_matches_generation(
                    user_feedback,
                    evolution_state['current_generation'],
                    expected_count,
                ):
                    print(
                        "Discarding queued feedback because its generation is no longer current."
                    )
                    continue

               
                try:
                    print("📦 Dequeued user_feedback object:")
                    print(json.dumps(user_feedback, indent=2, ensure_ascii=False)[:5000])
                except TypeError:
                    print(f"📦 Dequeued user_feedback (non-serializable): {str(user_feedback)[:5000]}")

                print(f"\n🎯 USER FEEDBACK RECEIVED!")
                print(f"Action: {user_feedback.get('action', 'unknown')}")
                print(f"Type: {user_feedback.get('type', 'unknown')}")
                print(f"Current tab: {user_feedback.get('current_tab', 'unknown')}")
                print(f"Total solutions: {user_feedback.get('total_solutions', 0)}")
                print(f"Actually scored: {user_feedback.get('actually_scored_solutions', 0)}")
                print(f"All solutions included: {user_feedback.get('all_solutions_included', False)}")
                
                # NEW VALIDATION: Check if feedback is complete
                actually_scored = user_feedback.get('actually_scored_solutions', 0)
                all_solutions_included = user_feedback.get('all_solutions_included', False)
                
                if not all_solutions_included:
                    print("❌ Incomplete feedback: not all solutions included, waiting for complete feedback...")
                    continue
                
                # Proceed with evolution - use user feedback if available, otherwise automatic fitness only
                evolution_state['waiting_for_feedback'] = False
                evolution_state['running'] = True
                
                if actually_scored > 0:
                    print(f"✅ Valid feedback received: {actually_scored} solutions scored")
                    print(f"🧬 STARTING EVOLUTION WITH USER FEEDBACK...")
                else:
                    print(f"⚠️ No solutions scored by user - using automatic fitness only")
                    print(f"🧬 STARTING EVOLUTION WITH AUTOMATIC FITNESS...")
                
                # Run evolution cycle (will use user feedback if available, auto fitness otherwise)
                run_evolution_cycle(user_feedback=user_feedback)
                
            except queue.Empty:
                # No feedback received, just continue waiting
                continue
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error in main loop: {e}")
    finally:
        print("🔚 Interactive evolution system stopped.")

if __name__ == "__main__":
    main()
