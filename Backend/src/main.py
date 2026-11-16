import os
import numpy as np
import json
from dataset_class import analyze_folder
import soundfile as sf
import matplotlib.pyplot as plt
import sounddevice as sd
from evolutionary_module import archive, emitters, automatic_fitness, general_bounds
from ribs.schedulers import Scheduler
from audio_synthesis import generate_audio_stream, extract_audio_descriptors
from utils import load_audio_catalog
import librosa
import concurrent.futures
import pandas as pd
import random
import multiprocessing as mp
import asyncio
import websockets
import threading
import base64
import io
import time
import queue


TRACKS_PER_SOLUTION = 2  # Number of tracks per solution
categories = ['low', 'mid', 'high', 'rhythmic']

# Global dictionaries to store track assignments
SOLUTION_TRACK_ASSIGNMENTS = {}  # Maps solution_id -> [track_indices]
SOLUTION_CATEGORY_ASSIGNMENTS = {}  # Maps solution_id -> assigned_category

# WebSocket server globals
websocket_clients = set()
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
    """Extract track paths from solution's assigned tracks using global dict."""
    solution_id = solution_to_id(solution)
    
    if solution_id in SOLUTION_TRACK_ASSIGNMENTS:
        track_indices = SOLUTION_TRACK_ASSIGNMENTS[solution_id]
    else:
        # Fallback to random tracks
        track_indices = [np.random.randint(0, len(catalog_df)) for _ in range(TRACKS_PER_SOLUTION)]
        SOLUTION_TRACK_ASSIGNMENTS[solution_id] = track_indices
    
    # Ensure valid indices
    track_indices = [np.clip(idx, 0, len(catalog_df) - 1) for idx in track_indices]
    
    # Get file paths
    track_paths = []
    for idx in track_indices:
        track_paths.append(catalog_df.iloc[idx]['file_path'])
    
    return track_paths

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

def get_assigned_tracks(solution):
    """Get assigned track indices for a solution."""
    solution_id = solution_to_id(solution)
    return SOLUTION_TRACK_ASSIGNMENTS.get(solution_id, [])



def _job(job):
    """Process a synthesis job """
    solution = job['solution']  
    catalog_data = job['catalog_data'] 

    catalog_df = pd.DataFrame(catalog_data)
    
    # Extract only audio parameters 
    audio_params = solution
    
    # Extract track paths from solution's assigned tracks
    source_tracks = get_tracks_from_solution(solution, catalog_df)
    
    # Synthesize audio using the provided data 
    audio_stream, sample_rate = generate_audio_stream(audio_params, duration=30, sources=source_tracks)
    
    descriptors = extract_audio_descriptors(audio_stream, sample_rate)

    
    return audio_stream, sample_rate, descriptors, source_tracks

def assign_category_from_descriptors(descriptors):
   
    centroid, contrast, spectral_flux, onset_rate, zcr = descriptors  

    
    if spectral_flux < 1500 and spectral_flux > 700 and onset_rate > 10.0 and zcr > 0.1: 
        return 'rhythmic'
    elif centroid <= 1200:
        return 'low'
    elif centroid >= 2500:
        return 'high'
    else: 
        return 'mid'

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

def calculate_hybrid_fitness(descriptors, category, user_score=None, was_scored=False, original_fitness=None):
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
        return ""

def start_websocket_server():
    """Start WebSocket server in a separate thread.""" 
    def run_server():
        async def websocket_handler(websocket):
            print(f"New WebSocket client connected from {websocket.remote_address}")
            websocket_clients.add(websocket)
            
            try:
                # Send ONLY the most recent generation data to new client
                if generation_data_latest is not None:
                    await websocket.send(json.dumps(generation_data_latest))
                    print(f"Sent most recent generation {generation_data_latest.get('generation', '?')} to new client")
                else:
                    print("No generation data available yet for new client")
                
                # Handle incoming messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        print(f"📨 Received message: {data.get('action', 'unknown')}")
                        # DEBUG: print full incoming payload
                        try:
                            print("🔍 Incoming payload (raw):")
                            print(json.dumps(data, indent=2, ensure_ascii=False)[:5000])
                        except Exception:
                            print(f"🔍 Incoming payload (raw str): {message[:5000]}")
                        
                        if data.get('action') == 'user_feedback' and data.get('type') == 'user_scoring':
                            print(f"🎯 User feedback received: {data.get('scored_solutions', 0)} solutions scored")
                            
                            try:
                                print("🧾 Enqueuing user_feedback:")
                                print(json.dumps(data, indent=2, ensure_ascii=False)[:5000])
                            except Exception:
                                pass
                            user_feedback_queue.put(data)
                        else:
                            print(f"❓ Unknown message type: {data}")
                    except json.JSONDecodeError:
                        print(f"❌ Invalid JSON received: {message}")
                    except Exception as e:
                        print(f"❌ Error processing message: {e}")
                    
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket client disconnected")
            except Exception as e:
                print(f"WebSocket error: {e}")
            finally:
                websocket_clients.discard(websocket)
        
        # FIXED: Create server inside async function
        async def start_server_async():
            server = await websockets.serve(websocket_handler, "localhost", 8765)
            print("✅ WebSocket server started on ws://localhost:8765")
            await asyncio.Future()  # Run forever
        
        # Create event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async server
        try:
            loop.run_until_complete(start_server_async())
        except KeyboardInterrupt:
            print("🛑 WebSocket server stopped")
        finally:
            loop.close()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    return server_thread

def send_generation_data(generation_num, offspring, results, objective_batch, categories_list, inheritor_indices):
    """Prepare generation data and queue it for WebSocket transmission."""
    global generation_data_latest  # Access the global variable
    
    print(f"🐍 PREPARING GENERATION {generation_num} DATA FOR WEBSOCKET...")
    
    generation_data = {
        "type": "generation_data",
        "generation": generation_num,
        "timestamp": int(time.time() * 1000),
        "solutions": []
    }
    
    print(f"🐍 Processing {len(offspring)} solutions...")
    
    for j, (sol, (audio_stream, sample_rate, descriptors, source_tracks)) in enumerate(zip(offspring, results)):
        assigned_category = get_assigned_category(sol)
        actual_category = categories_list[j]
        fitness = float(objective_batch[j])
        is_inherited = j in inheritor_indices
        
        print(f"🐍 Converting audio for solution {j+1}/{len(offspring)}")
        
        # Convert audio to base64
        try:
            audio_base64 = audio_to_base64(audio_stream, sample_rate)
            print(f"  ✅ Audio converted: {len(audio_base64)} chars")
        except Exception as e:
            print(f"  ❌ Audio conversion failed: {e}")
            audio_base64 = ""
        
        solution_data = {
            "id": j,
            "solution_id": f"solution_{generation_num:03d}_{j:03d}",  # Unique solution ID
            "parameters": {
                "grain_duration": float(sol[0]),
                "overlap": float(sol[1]),
                "pitch_variance": float(sol[2]),
                "start_offset_range": float(sol[3]),
                "room_size": float(sol[4]),
                "damping": float(sol[5]),
                "wet_level": float(sol[6]),
                "num_voices": int(sol[7]),
                "low_cutoff_hz": float(sol[8]),
                "high_cutoff_hz": float(sol[9]),
                "ladder_resonance": float(sol[10])
            },
            "descriptors": {
                "centroid": float(descriptors[0]),
                "rolloff": float(descriptors[1]),
                "spectral_flux": float(descriptors[2]),
                "onset_rate": float(descriptors[3]),
                "zcr": float(descriptors[4]) if len(descriptors) > 4 else 0.0
            },
            "fitness": fitness,
            "assigned_category": assigned_category,
            "actual_category": actual_category,
            "inherited": is_inherited,
            "sources": [os.path.basename(path) for path in source_tracks],
            "audio": {
                "data": audio_base64,
                "sample_rate": int(sample_rate),
                "duration": len(audio_stream) / sample_rate
            }
        }
        
        generation_data["solutions"].append(solution_data)
        print(f"  ✅ Solution {j+1} data prepared")
    
    # Add generation statistics
    generation_data["statistics"] = {
        "total_solutions": len(offspring),
        "inherited_count": len(inheritor_indices),
        "new_count": len(offspring) - len(inheritor_indices),
        "avg_fitness": float(np.mean(objective_batch)),
        "best_fitness": float(np.max(objective_batch)),
        "worst_fitness": float(np.min(objective_batch)),
        "category_distribution": {
            category: categories_list.count(category) 
            for category in ['low', 'mid', 'high', 'rhythmic']
        }
    }
    
    # Store ONLY the latest generation (replaces previous)
    generation_data_latest = generation_data
    print(f"🐍 GENERATION {generation_num} DATA STORED AS LATEST")
    print(f"  - Solutions: {len(generation_data['solutions'])}")
    print(f"  - Connected clients: {len(websocket_clients)}")
    
    # Broadcast immediately to connected clients
    if websocket_clients:
        print(f"🐍 BROADCASTING TO {len(websocket_clients)} CLIENTS...")
        
        # Create a simple broadcast function that works
        def broadcast_now():
            import asyncio
            import json
            
            async def do_broadcast():
                message = json.dumps(generation_data)
                disconnected = set()
                
                for client in websocket_clients.copy():  # Use copy to avoid modification during iteration
                    try:
                        await client.send(message)
                        print(f"  ✅ Sent to client {client.remote_address}")
                    except Exception as e:
                        print(f"  ❌ Failed to send to client: {e}")
                        disconnected.add(client)
                
                # Remove disconnected clients
                for client in disconnected:
                    websocket_clients.discard(client)
                
                print(f"🐍 BROADCAST COMPLETE: {len(websocket_clients)} clients remain")
            
            # Get the event loop from the WebSocket server thread
            try:
                # Use asyncio.run for a clean async execution
                asyncio.run(do_broadcast())
                
            except Exception as e:
                print(f"🐍 BROADCAST ERROR: {e}")
        
        # Run broadcast in a separate thread to avoid blocking
        broadcast_thread = threading.Thread(target=broadcast_now)
        broadcast_thread.start()
    
    else:
        print("🐍 NO CLIENTS CONNECTED - LATEST DATA STORED FOR FUTURE CONNECTIONS")
    
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

    # Synthesize
    synthesis_jobs = []
    for sol in offspring:
        synthesis_jobs.append({
            'solution': sol,
            'catalog_data': catalog_df.to_dict('records')
        })

    results = []
    max_workers = min(mp.cpu_count(), len(synthesis_jobs))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_job, job) for job in synthesis_jobs]
        for k, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error: {e}")
                fallback_audio = np.zeros(48000)
                fallback_sr = 48000
                fallback_descriptors = np.zeros(5)
                fallback_sources = ["fallback_track"]
                results.append((fallback_audio, fallback_sr, fallback_descriptors, fallback_sources))

    # Calculate automatic fitness
    objective_batch = []
    categories_list = []

    for j, (sol, (audio_stream, sample_rate, descriptors, source_tracks)) in enumerate(zip(offspring, results)):
        actual_category = assign_category_from_descriptors(descriptors)
        fitness = automatic_fitness(descriptors, actual_category)
        objective_batch.append(fitness)
        categories_list.append(actual_category)

    objective_batch = np.array(objective_batch)

    # Select best 8
    selected_indices = select_best_solutions(offspring, results, objective_batch, categories_list, n_select=8)

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
def save_best_solutions(offspring, results, objective_batch, generation_num, save_dir="../data"):
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


def run_evolution_cycle(user_feedback=None, num_generations=1):
    """Run one evolution cycle - generate ONE generation and wait for feedback."""
    
    scheduler = evolution_state['scheduler']
    catalog_df = evolution_state['catalog_df']
    
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
                    # Extract the sent position from solution_id (the last 3 digits)
                    # solution_002_003 -> sent_position = 3
                    try:
                        sent_position = int(solution_id.split('_')[-1])
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
        save_best_solutions(offspring_selected, results_selected, objective_batch, evolution_state['current_generation'], save_dir="../data")
        
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
    
    # Synthesize solutions
    synthesis_jobs = []
    for sol in offspring:
        synthesis_jobs.append({
            'solution': sol,
            'catalog_data': catalog_df.to_dict('records')
        })
    
    results = []
    print("Starting synthesis (multiprocessing)")
    max_workers = min(mp.cpu_count(), len(synthesis_jobs))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_job, job) for job in synthesis_jobs]
        
        for k, future in enumerate(futures):
            try:
                result = future.result()  
                results.append(result)
                print(f"Completed job {k+1}/{len(futures)}")
            except Exception as e:
                print(f"Error processing job {k}: {e}")
                fallback_audio = np.zeros(48000)
                fallback_sr = 48000
                fallback_descriptors = np.zeros(5)
                fallback_sources = ["fallback_track"]
                results.append((fallback_audio, fallback_sr, fallback_descriptors, fallback_sources))
    
    # Calculate AUTOMATIC fitness only (no user feedback yet for THIS generation)
    objective_batch = []
    categories_list = []
    
    print(f"\n--- Generation {generation_num} Solutions (awaiting feedback) ---")
    
    for j, (sol, (audio_stream, sample_rate, descriptors, source_tracks)) in enumerate(zip(offspring, results)):
        actual_category = assign_category_from_descriptors(descriptors)
        fitness = automatic_fitness(descriptors, actual_category)
        
        objective_batch.append(fitness)
        categories_list.append(actual_category)
        
        print(f"Solution {j+1}: category={actual_category}, auto_fitness={fitness:.3f}")
    
    objective_batch = np.array(objective_batch)
    
# STEP 5: Select best 8 first (needed for storing mapping)
    selected_indices = select_best_solutions(offspring, results, objective_batch, categories_list, n_select=8)

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

def select_best_solutions(offspring, results, objective_batch, categories_list, n_select=8):
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
        catalog_df = pd.read_csv("../data/audio_analysis.csv")
        print(f"Loaded audio catalog with {len(catalog_df)} tracks")
        print(f"Category distribution: {catalog_df['dominant_category'].value_counts().to_dict()}")
        
        scheduler = Scheduler(archive, emitters)
        evolution_state['scheduler'] = scheduler
        evolution_state['catalog_df'] = catalog_df
        
    except Exception as e:
        print(f"Error loading audio catalog: {e}")
        print("Running analyze_folder to create catalog...")
        catalog_df = analyze_folder("../data/audio", "../data/audio_analysis.csv")
        
        scheduler = Scheduler(archive, emitters)
        evolution_state['scheduler'] = scheduler
        evolution_state['catalog_df'] = catalog_df
    
    # Create data directory
    data_dir = '../data'
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
                run_evolution_cycle(user_feedback=user_feedback, num_generations=1)
                
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



