#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for testing reward-guided protein denoising strategies.
"""

import os
import sys
import numpy as np
import random
import torch
from datetime import datetime
import argparse

from esm.sdk.api import ESMProtein, ESM3InferenceClient
from denoising_strategies import (
    RewardGuidedBaseDenoisingStrategy,
    EntropyBasedRewardGuidedDenoising,
    MaxProbBasedRewardGuidedDenoising,
    SimulatedAnnealingRewardGuidedDenoising,
    Tee
)

# Define some example reward functions for protein optimization

def hydrophobicity_reward(protein: ESMProtein) -> float:
    """
    Calculate a hydrophobicity score for the protein.
    Higher values indicate more hydrophobic residues.
    
    This is a simple example that rewards proteins with more hydrophobic amino acids.
    
    Args:
        protein: The protein to evaluate
        
    Returns:
        float: Hydrophobicity score
    """
    # Kyte & Doolittle hydrophobicity scale
    hydrophobicity_scale = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    
    # Calculate average hydrophobicity
    sequence = protein.sequence
    hydrophobicity_sum = sum(hydrophobicity_scale.get(aa, 0) for aa in sequence)
    return hydrophobicity_sum / len(sequence) if sequence else 0


def charge_reward(protein: ESMProtein) -> float:
    """
    Calculate net charge of the protein at physiological pH.
    
    This example reward function aims for a neutral net charge.
    
    Args:
        protein: The protein to evaluate
        
    Returns:
        float: Negative absolute charge (higher is better, with 0 being optimal)
    """
    # Amino acid charges at physiological pH
    charges = {
        'D': -1, 'E': -1,  # Negatively charged
        'K': 1, 'R': 1, 'H': 0.1  # Positively charged
    }
    
    sequence = protein.sequence
    total_charge = sum(charges.get(aa, 0) for aa in sequence)
    
    # Return negative absolute charge (closer to 0 is better)
    return -abs(total_charge)


def secondary_structure_reward(protein: ESMProtein) -> float:
    """
    A simplified secondary structure propensity reward.
    This example rewards sequences with amino acids that favor alpha helices.
    
    Args:
        protein: The protein to evaluate
        
    Returns:
        float: Alpha helix propensity score
    """
    # Alpha helix propensities (higher values favor alpha helices)
    # Based on Chou-Fasman parameters
    helix_propensity = {
        'A': 1.42, 'C': 0.70, 'D': 1.01, 'E': 1.51, 'F': 1.13,
        'G': 0.57, 'H': 1.00, 'I': 1.08, 'K': 1.16, 'L': 1.21,
        'M': 1.45, 'N': 0.67, 'P': 0.57, 'Q': 1.11, 'R': 0.98,
        'S': 0.77, 'T': 0.83, 'V': 1.06, 'W': 1.08, 'Y': 0.69
    }
    
    sequence = protein.sequence
    # Calculate average helix propensity
    propensity_sum = sum(helix_propensity.get(aa, 0) for aa in sequence)
    return propensity_sum / len(sequence) if sequence else 0


def combined_reward(protein: ESMProtein) -> float:
    """
    A combined reward function that considers multiple properties.
    
    Args:
        protein: The protein to evaluate
        
    Returns:
        float: Combined reward score
    """
    hydrophobicity = hydrophobicity_reward(protein)
    charge = charge_reward(protein)
    structure = secondary_structure_reward(protein)
    
    # Normalize and combine with weights
    return 0.4 * hydrophobicity + 0.3 * (charge + 5) + 0.3 * structure


def main():
    parser = argparse.ArgumentParser(description='Test reward-guided protein denoising strategies')
    parser.add_argument('--strategy', type=str, default='entropy',
                        choices=['entropy', 'maxprob', 'annealing'],
                        help='Denoising strategy to use')
    parser.add_argument('--reward', type=str, default='combined',
                        choices=['hydrophobicity', 'charge', 'structure', 'combined'],
                        help='Reward function to use')
    parser.add_argument('--lookahead', type=int, default=1,
                        help='Number of lookahead steps')
    parser.add_argument('--protein', type=str, default=None,
                        help='Path to PDB file (if not provided, will use a default sequence)')
    parser.add_argument('--noise', type=float, default=50.0,
                        help='Percentage of positions to mask')
    parser.add_argument('--steps', type=int, default=20,
                        help='Number of denoising steps')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Sampling temperature (or base temperature for simulated annealing)')
                        
    args = parser.parse_args()
    
    # Setup output file for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"denoising_outputs/denoising_output_{timestamp}.txt"
    os.makedirs("denoising_outputs", exist_ok=True)
    
    sys.stdout = Tee(output_file)
    
    print(f"Starting reward-guided protein denoising test with {args.strategy} strategy")
    print(f"Using {args.reward} reward function with {args.lookahead} lookahead steps")
    
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Initialize ESM client
    client = ESM3InferenceClient()
    print("Initialized ESM3 client")
    
    # Select reward function
    reward_functions = {
        'hydrophobicity': hydrophobicity_reward,
        'charge': charge_reward,
        'structure': secondary_structure_reward,
        'combined': combined_reward
    }
    reward_function = reward_functions[args.reward]
    
    # Create protein (either from file or default sequence)
    if args.protein:
        print(f"Loading protein from {args.protein}")
        protein = ESMProtein.from_pdb_file(args.protein)
    else:
        # Simple example sequence
        default_sequence = "MASKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
        print(f"Using default sequence: {default_sequence}")
        protein = ESMProtein(default_sequence)
    
    print(f"Protein length: {len(protein.sequence)}")
    
    # Initialize strategy based on selection
    strategy_params = {
        'client': client,
        'noise_percentage': args.noise,
        'num_decoding_steps': args.steps,
        'reward_function': reward_function,
        'lookahead_steps': args.lookahead
    }
    
    if args.strategy == 'entropy':
        strategy_params['temperature'] = args.temperature
        strategy = EntropyBasedRewardGuidedDenoising(**strategy_params)
    elif args.strategy == 'maxprob':
        strategy_params['temperature'] = args.temperature
        strategy = MaxProbBasedRewardGuidedDenoising(**strategy_params)
    elif args.strategy == 'annealing':
        strategy_params['base_temperature'] = args.temperature
        strategy = SimulatedAnnealingRewardGuidedDenoising(**strategy_params)
    
    # Run denoising
    print("Starting denoising process...")
    denoised_protein = strategy.denoise(protein, verbose=True)
    
    # Print results
    print("\n=== Results ===")
    print(f"Original protein: {protein.sequence}")
    print(f"Denoised protein: {denoised_protein.sequence}")
    print(f"Original reward: {reward_function(protein):.4f}")
    print(f"Denoised reward: {reward_function(denoised_protein):.4f}")
    print(f"Total model calls: {strategy.cost}")
    print("=== End of Results ===\n")
    
    # Return to normal stdout
    if isinstance(sys.stdout, Tee):
        sys.stdout.close()
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()