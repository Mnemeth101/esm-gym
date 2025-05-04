#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark script to compare reward-guided denoising strategies with regular denoising strategies.
"""

import os
import sys
import json
import numpy as np
import random
import torch
from datetime import datetime
import argparse
import matplotlib.pyplot as plt

from esm.sdk.api import ESMProtein, ESM3InferenceClient
from denoising_strategies import (
    EntropyBasedDenoising,
    MaxProbBasedDenoising,
    SimulatedAnnealingDenoising,
    RewardGuidedBaseDenoisingStrategy,
    EntropyBasedRewardGuidedDenoising,
    MaxProbBasedRewardGuidedDenoising,
    SimulatedAnnealingRewardGuidedDenoising,
)
from denoising_strategies import Tee

# Import reward functions
from benchmarking_utils import benchmark_protein

# Import reward functions we defined in the testing script
from _4_RewardGuidedDenoising_testing import (
    hydrophobicity_reward,
    charge_reward,
    secondary_structure_reward,
    combined_reward
)

from benchmarking import BenchmarkRunner, load_protein_from_fasta

def run_benchmark(args):
    """Run benchmark comparing standard and reward-guided strategies."""
    # Setup output directory for benchmarking results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = f"data/benchmarking_results/RewardGuidedDenoising_{timestamp}"
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Output log file
    log_file = os.path.join(benchmark_dir, "benchmark_log.txt")
    sys.stdout = Tee(log_file)
    
    print(f"Starting benchmark for reward-guided denoising strategies")
    print(f"Strategy type: {args.strategy}")
    print(f"Reward function: {args.reward}")
    print(f"Lookahead steps: {args.lookahead}")
    
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
    
    # Load protein
    if args.protein:
        print(f"Loading protein from {args.protein}")
        try:
            protein = ESMProtein.from_pdb_file(args.protein)
            print(f"Loaded protein with sequence length: {len(protein.sequence)}")
        except Exception as e:
            print(f"Error loading protein: {e}")
            return
    else:
        # Simple example sequence (Green Fluorescent Protein)
        default_sequence = "MASKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
        print(f"Using default sequence: {default_sequence}")
        protein = ESMProtein(default_sequence)

    # Parameters for all strategies
    base_params = {
        'client': client,
        'noise_percentage': args.noise,
        'num_decoding_steps': args.steps,
    }
    
    # Create pairs of standard and reward-guided strategies for comparison
    if args.strategy == 'entropy':
        base_params['temperature'] = args.temperature
        standard_strategy = EntropyBasedDenoising(**base_params)
        reward_params = base_params.copy()
        reward_params.update({
            'reward_function': reward_function,
            'lookahead_steps': args.lookahead
        })
        reward_guided_strategy = EntropyBasedRewardGuidedDenoising(**reward_params)
    elif args.strategy == 'maxprob':
        base_params['temperature'] = args.temperature
        standard_strategy = MaxProbBasedDenoising(**base_params)
        reward_params = base_params.copy()
        reward_params.update({
            'reward_function': reward_function,
            'lookahead_steps': args.lookahead
        })
        reward_guided_strategy = MaxProbBasedRewardGuidedDenoising(**reward_params)
    elif args.strategy == 'annealing':
        base_params['base_temperature'] = args.temperature
        standard_strategy = SimulatedAnnealingDenoising(**base_params)
        reward_params = base_params.copy()
        reward_params.update({
            'reward_function': reward_function,
            'lookahead_steps': args.lookahead
        })
        reward_guided_strategy = SimulatedAnnealingRewardGuidedDenoising(**reward_params)
    
    # Run standard strategy
    print("\n=== Running Standard Strategy ===")
    standard_start_time = datetime.now()
    standard_protein = standard_strategy.denoise(protein, verbose=False)
    standard_end_time = datetime.now()
    standard_duration = (standard_end_time - standard_start_time).total_seconds()
    standard_cost = standard_strategy.cost
    standard_reward = reward_function(standard_protein)
    
    # Run reward-guided strategy
    print("\n=== Running Reward-Guided Strategy ===")
    reward_start_time = datetime.now()
    reward_protein = reward_guided_strategy.denoise(protein, verbose=False)
    reward_end_time = datetime.now()
    reward_duration = (reward_end_time - reward_start_time).total_seconds()
    reward_cost = reward_guided_strategy.cost
    reward_guided_reward = reward_function(reward_protein)
    
    # Print comparison results
    print("\n=== Benchmark Results ===")
    print(f"Original protein reward: {reward_function(protein):.4f}")
    print("\nStandard Strategy:")
    print(f"- Final reward: {standard_reward:.4f}")
    print(f"- Model calls: {standard_cost}")
    print(f"- Duration: {standard_duration:.2f} seconds")
    
    print("\nReward-Guided Strategy:")
    print(f"- Final reward: {reward_guided_reward:.4f}")
    print(f"- Model calls: {reward_cost}")
    print(f"- Duration: {reward_duration:.2f} seconds")
    
    print(f"\nImprovement with reward guidance: {(reward_guided_reward - standard_reward):.4f} ({((reward_guided_reward - standard_reward) / abs(standard_reward) * 100):.2f}%)")
    print(f"Additional model calls: {reward_cost - standard_cost} ({(reward_cost / standard_cost):.2f}x)")
    
    # Save results to JSON
    results = {
        "benchmark_settings": {
            "strategy": args.strategy,
            "reward_function": args.reward,
            "lookahead_steps": args.lookahead,
            "noise_percentage": args.noise,
            "num_steps": args.steps,
            "temperature": args.temperature,
            "protein_path": args.protein,
            "protein_length": len(protein.sequence)
        },
        "standard_strategy": {
            "reward": float(standard_reward),
            "model_calls": standard_cost,
            "duration_seconds": standard_duration
        },
        "reward_guided_strategy": {
            "reward": float(reward_guided_reward),
            "model_calls": reward_cost,
            "duration_seconds": reward_duration
        },
        "improvement": {
            "absolute": float(reward_guided_reward - standard_reward),
            "percentage": float((reward_guided_reward - standard_reward) / abs(standard_reward) * 100),
            "cost_ratio": float(reward_cost / standard_cost)
        }
    }
    
    # Save results to JSON file
    results_file = os.path.join(benchmark_dir, "benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    # Create visualization of results
    create_benchmark_visualization(results, benchmark_dir)
    
    print(f"\nResults saved to {benchmark_dir}")
    
    # Return to normal stdout
    if isinstance(sys.stdout, Tee):
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        
    return results

def create_benchmark_visualization(results, output_dir):
    """Create visualizations for benchmark results."""
    # Create bar chart comparing rewards
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = ['Standard', 'Reward-Guided']
    rewards = [
        results['standard_strategy']['reward'],
        results['reward_guided_strategy']['reward']
    ]
    
    # Get strategy type and reward function for title
    strategy_type = results['benchmark_settings']['strategy']
    reward_function = results['benchmark_settings']['reward']
    
    # Create bars
    bars = ax.bar(strategies, rewards, color=['lightblue', 'darkblue'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Add improvement percentage
    improvement = results['improvement']['percentage']
    ax.annotate(f'+{improvement:.2f}%', 
                xy=(1, rewards[1]), 
                xytext=(1.2, rewards[1] * 0.9),
                arrowprops=dict(arrowstyle='->'))
    
    # Set labels and title
    ax.set_ylabel('Reward Value')
    ax.set_title(f'Reward Comparison: {strategy_type.capitalize()}-Based Strategies\nReward Function: {reward_function}')
    
    # Add cost information in text box
    std_calls = results['standard_strategy']['model_calls']
    reward_calls = results['reward_guided_strategy']['model_calls']
    cost_ratio = results['improvement']['cost_ratio']
    
    textstr = f'Model Calls:\nStandard: {std_calls}\nReward-Guided: {reward_calls}\nRatio: {cost_ratio:.2f}x'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Save figure
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_comparison.png'))
    
    # Create second plot for model calls
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Model calls bar chart
    calls = [std_calls, reward_calls]
    bars2 = ax2.bar(strategies, calls, color=['lightgreen', 'darkgreen'])
    
    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Set labels and title
    ax2.set_ylabel('Number of Model Calls')
    ax2.set_title(f'Efficiency Comparison: {strategy_type.capitalize()}-Based Strategies\nReward Function: {reward_function}')
    
    # Add cost ratio annotation
    ax2.annotate(f'{cost_ratio:.2f}x', 
                xy=(1, calls[1]), 
                xytext=(1.2, calls[1] * 0.8),
                arrowprops=dict(arrowstyle='->'))
    
    # Save figure
    fig2.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calls_comparison.png'))

def main():
    parser = argparse.ArgumentParser(description='Benchmark reward-guided protein denoising strategies')
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
    run_benchmark(args)

if __name__ == "__main__":
    main()