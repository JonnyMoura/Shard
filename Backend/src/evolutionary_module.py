import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter, GeneticAlgorithmEmitter, IsoLineEmitter
from ribs.schedulers import Scheduler
import soundfile as sf
import os
import json
from audio_synthesis import generate_audio_stream
import sounddevice as sd
from ribs.emitters.operators import GaussianOperator


def audio_fitness_interactive(params):
    """Interactive fitness function where the user ranks each solution."""
    while True:
        try:
            fitness = float(input("Rate this solution on a scale from 1 to 100: "))
            if 1 <= fitness <= 100:
                break
            else:
                print("Please enter a number between 1 and 100.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 100.")
    return fitness

def automatic_fitness(descriptors, category=None):
    
    centroid, rolloff, spectral_flux, onset_rate, zcr = descriptors  

    def balanced_target_fitness(value, target_min, target_max, power=2.0):
        
        if target_min <= value <= target_max:
            
            center = (target_min + target_max) / 2
            range_width = target_max - target_min
            distance_from_center = abs(value - center) / (range_width / 2)
            score = (1.0 - distance_from_center) ** power
            return 0.6 + 0.4 * score  
        else:
            
            if value < target_min:
                distance = (target_min - value) / target_min
            else:
                distance = (value - target_max) / target_max
            
           
            decay_score = np.exp(-2.5 * distance)
            return 0.3 + 0.3 * decay_score  

    def frequency_balance_fitness(rolloff, category_type, power=2.0):
        
        if category_type == "low":
            # Low sounds have lower rolloff (energy concentrated in bass)
            return balanced_target_fitness(rolloff, 800, 2000, power)
        elif category_type == "mid":
            # Mid sounds have moderate rolloff (balanced energy)
            return balanced_target_fitness(rolloff, 2000, 4000, power)
        elif category_type == "high":
            # High sounds have higher rolloff (energy extends to highs)
            return balanced_target_fitness(rolloff, 3000, 5000, power)
        else:  # rhythmic
            # Rhythmic sounds vary widely based on drum content
            return balanced_target_fitness(rolloff, 1800, 4500, power)

    def zcr_fitness(zcr, category_type, power=1.8):
        
        if category_type == "rhythmic":
            # Rhythmic sounds - high zcr
            return balanced_target_fitness(zcr, 0.05, 0.3, power)  
        elif category_type == "low":        
            return balanced_target_fitness(zcr, 0.005, 0.12, power)  
        elif category_type == "mid":
         
            return balanced_target_fitness(zcr, 0.01, 0.15, power)  
        else:  
         
            return balanced_target_fitness(zcr, 0.02, 0.18, power) 

    def continuity_fitness(spectral_flux, onset_rate):
        
        flux_score = balanced_target_fitness(spectral_flux, 800, 3000, 2.0)  
        
        if onset_rate <= 4.0:
            onset_score = 0.7 + 0.2 * (onset_rate / 2.0)  
        elif onset_rate <= 8.0:
            onset_score = 0.80  
        else:
            onset_score = 0.7 - 0.1 * min(1.0, (onset_rate - 10.0) / 10.0)  
        
        return (flux_score + onset_score) / 2

    def rhythmic_fitness(spectral_flux, onset_rate):
        
       
        if 700 <= spectral_flux <= 2500:  
            flux_score = 0.7 + 0.3 * (1.0 - abs(spectral_flux - 850) / 650)
        elif spectral_flux < 200:
            flux_score = 0.5 + 0.2 * (spectral_flux / 200)
        else:  # spectral_flux > 1500
            overshoot = (spectral_flux - 1500) / 2000 
            flux_score = max(0.3, 0.7 * np.exp(-2.0 * overshoot))  
        
        if onset_rate >= 9.0:  
            if onset_rate <= 15.0:  
                onset_score = 0.6 + 0.4 * (onset_rate - 4.0) / 16.0
            else:
                onset_score = 1.0 - 0.1 * ((onset_rate - 20.0) / 10.0) 
        elif onset_rate >= 2.0:
            onset_score = 0.4 + 0.2 * (onset_rate - 2.0) / 2.0
        else:
            onset_score = max(0.1, 0.4 * (onset_rate / 2.0)) 
        
        return (flux_score + onset_score) / 2

    def balanced_fitness_multiplier(individual_scores):
       
        avg_score = np.mean(individual_scores)
        min_score = np.min(individual_scores)
        
       
        if min_score < 0.4:
            balance_penalty = 0.7 + 0.3 * (min_score / 0.4) 
        else:
            balance_penalty = 1.0 + 0.2 * (min_score - 0.4) / 0.6  
        
       
        if avg_score > 0.7:  
            excellence_bonus = 1.0 + 0.3 * (avg_score - 0.7) / 0.3
        elif avg_score > 0.5:  
            excellence_bonus = 1.0 + 0.1 * (avg_score - 0.5) / 0.2
        else:
            excellence_bonus = 0.9 + 0.1 * (avg_score / 0.5) 
        
        return min(1.6, balance_penalty * excellence_bonus)  
    
   
    if category == "low":
        centroid_score = balanced_target_fitness(centroid, 400, 1800, 2.0)  
        continuity_score = continuity_fitness(spectral_flux, onset_rate)
        balance_score = frequency_balance_fitness(rolloff, "low")  
        zcr_score = zcr_fitness(zcr, "low", 1.8)
        
        scores = [centroid_score, continuity_score, balance_score, zcr_score]
        base_fitness = 0.45 * centroid_score + 0.2 * continuity_score + 0.25 * balance_score + 0.10 * zcr_score

    elif category == "mid":
        centroid_score = balanced_target_fitness(centroid, 1200, 3000, 2.5) 
        continuity_score = continuity_fitness(spectral_flux, onset_rate)
        balance_score = frequency_balance_fitness(rolloff, "mid")  
        zcr_score = zcr_fitness(zcr, "mid", 1.8)
        
        scores = [centroid_score, continuity_score, balance_score, zcr_score]
        base_fitness = 0.4 * centroid_score + 0.2 * continuity_score + 0.25 * balance_score + 0.15 * zcr_score
        
    elif category == "high":
        centroid_score = balanced_target_fitness(centroid, 2500, 3500, 2.5)  
        continuity_score = continuity_fitness(spectral_flux, onset_rate)
        balance_score = frequency_balance_fitness(rolloff, "high")  # Category-specific
        zcr_score = zcr_fitness(zcr, "high", 1.8)
        
        scores = [centroid_score, continuity_score, balance_score, zcr_score]
        base_fitness = 0.4 * centroid_score + 0.2 * continuity_score + 0.25 * balance_score + 0.15 * zcr_score

    elif category == "rhythmic":
        centroid_score = balanced_target_fitness(centroid, 800, 2000, 2.0) 
        rhythmic_score = rhythmic_fitness(spectral_flux, onset_rate)
        balance_score = frequency_balance_fitness(rolloff, "rhythmic")  # Category-specific
        zcr_score = zcr_fitness(zcr, "rhythmic", 2.0)
        
        scores = [centroid_score, rhythmic_score, balance_score, zcr_score]
        base_fitness = 0.25 * centroid_score + 0.35 * rhythmic_score + 0.15 * balance_score + 0.25 * zcr_score
    
    else:
        return 0.5 
    
   
    multiplier = balanced_fitness_multiplier(scores)
    final_fitness = base_fitness * multiplier
    
   
    scaled_fitness = 0.1 + 0.8 / (1.0 + np.exp(-4.0 * (final_fitness - 0.5))) 
    
    return min(1.0, max(0.1, scaled_fitness))




archive = GridArchive(
    solution_dim=11,  
    dims=[6, 6, 5],
    ranges=[
        (300, 5000),     # Centroid - frequency content
        (300, 4000),     # Spectral Flux - temporal patterns 
        (800, 6000),     # Spectral Rolloff - frequency balance
    ]
)

general_bounds = (
    np.array([0.1, 0.5, -0.05, 1000, 0.01, 0.1, 0.05, 2, 40, 400, 0.01]),   
    np.array([1.0, 10.0, 0.05, 8000000, 0.4, 0.4, 0.3, 8, 3000, 12000, 0.3])
)

def create_emitters():
    emitters = []
    
    for i in range(4):
        emitter_1 = EvolutionStrategyEmitter(
            archive, 
            x0=np.random.uniform(low=general_bounds[0], high=general_bounds[1]),
            sigma0=0.7, 
            batch_size=4
        )
        
        emitter_2 = IsoLineEmitter(
            archive,
            x0=np.random.uniform(low=general_bounds[0], high=general_bounds[1]),
            batch_size=4
        )
        emitter_3 = EvolutionStrategyEmitter(
            archive, 
            x0=np.random.uniform(low=general_bounds[0], high=general_bounds[1]),
            sigma0=0.7, 
            batch_size=4
        )
        emitter_4=EvolutionStrategyEmitter(
            archive, 
            x0=np.random.uniform(low=general_bounds[0], high=general_bounds[1]),
            sigma0=0.7, 
            batch_size=4

        )
        emitters.append(emitter_1)
        emitters.append(emitter_2)
        emitters.append(emitter_3)
        emitters.append(emitter_4)

    return emitters




emitters = create_emitters()





