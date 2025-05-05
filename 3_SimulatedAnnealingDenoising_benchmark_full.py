import os
import json
import torch
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple

from esm.sdk.api import ESM3InferenceClient
from esm.models.esm3 import ESM3

from denoising_strategies import SimulatedAnnealingDenoising
from utils import (
    load_proteins,
    calculate_metrics,
    save_results,
    setup_logging,
    get_esm_client
)

def run_benchmark(
    client: ESM3InferenceClient,
    proteins: List[Dict],
    output_dir: str,
    num_decoding_steps: int = 20,
    noise_percentage: float = 50.0,
    base_temperature: float = 1.0,
    schedule_type: str = 'linear', # Added schedule_type parameter
    num_samples: int = 5
) -> Dict:
    """Run benchmark for SimulatedAnnealingDenoising strategy."""
    
    # Initialize strategy
    strategy = SimulatedAnnealingDenoising(
        client=client,
        noise_percentage=noise_percentage,
        num_decoding_steps=num_decoding_steps,
        base_temperature=base_temperature,
        schedule_type=schedule_type # Pass schedule_type
    )
    
    # Initialize results
    results = {
        "metrics": [],
        "costs": [],
        "sequences": [],
        "temperatures": []  # Track temperature at each step
    }
    
    # Run benchmark for each protein
    for protein in tqdm(proteins, desc="Benchmarking proteins"):
        protein_id = protein["id"]
        print(f"\nProcessing protein {protein_id}")
        
        # Run multiple samples
        for sample in range(num_samples):
            print(f"Sample {sample + 1}/{num_samples}")
            
            # Run strategy
            decoded_protein, cost = strategy.denoise(protein["protein"])
            
            # Calculate metrics
            metrics = calculate_metrics(
                client=client,
                original_protein=protein["protein"],
                decoded_protein=decoded_protein
            )
            
            # Store results
            results["metrics"].append(metrics)
            results["costs"].append(cost)
            results["sequences"].append(decoded_protein.sequence)
            
            print(f"Cost: {cost}")
            print(f"Metrics: {metrics}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"simulated_annealing_benchmark_{timestamp}.json")
    
    # Convert results to serializable format
    serializable_results = {
        "metrics": [
            {
                "plddt": float(metrics["plddt"]),
                "ptm": float(metrics["ptm"]),
                "entropy": float(metrics["entropy"]),
                "diversity": float(metrics["diversity"]),
                "cosine_similarity": float(metrics["cosine_similarity"]),
                "unmask_entropy": float(metrics["unmask_entropy"])
            }
            for metrics in results["metrics"]
        ],
        "costs": results["costs"],
        "sequences": results["sequences"]
    }
    
    save_results(serializable_results, results_file)
    print(f"\nResults saved to {results_file}")
    
    return results

def main():
    # Setup
    output_dir = "results/simulated_annealing"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, "simulated_annealing_benchmark")
    
    # Load proteins
    proteins = load_proteins("data/casp15_single_domains.json")
    
    # Get ESM client
    client = get_esm_client()
    
    # Run benchmark
    results = run_benchmark(
        client=client,
        proteins=proteins,
        output_dir=output_dir,
        num_decoding_steps=20,
        noise_percentage=50.0,
        base_temperature=1.0,
        schedule_type='linear', # Pass schedule_type
        num_samples=5
    )
    
    # Print summary
    print("\nBenchmark Summary:")
    avg_metrics = {
        metric: sum(sample[metric] for sample in results["metrics"]) / len(results["metrics"])
        for metric in results["metrics"][0].keys()
    }
    avg_cost = sum(results["costs"]) / len(results["costs"])
    
    print("\nSimulatedAnnealingDenoising:")
    print(f"Average metrics: {avg_metrics}")
    print(f"Average cost: {avg_cost}")

if __name__ == "__main__":
    main()