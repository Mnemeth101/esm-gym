import os
import json
import time
from typing import List, Dict, Any, Optional
from esm.sdk.api import ESM3InferenceClient, ESMProtein, ESMProteinError, LogitsConfig
from esm.sdk.forge import ESM3ForgeInferenceClient
from .metrics import (
    single_metric_UACCE,
    single_metric_average_pLDDT,
    single_metric_pTM,
    single_metric_foldability,
    aggregated_metric_entropy,
    aggregated_metric_diversity,
    aggregated_cosine_similarities
)

class BenchmarkRunner:
    """
    A class to run benchmarks for protein denoising strategies.
    
    This class handles:
    1. Running multiple trials of a denoising strategy
    2. Collecting metrics for each trial
    3. Aggregating results across trials
    4. Saving results to a JSON file
    """
    
    def __init__(self, strategy_name: str, output_dir: str = "benchmark_results"):
        """
        Initialize the benchmark runner.
        
        Args:
            strategy_name: Name of the denoising strategy being benchmarked
            output_dir: Directory to save benchmark results
        """
        self.strategy_name = strategy_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Get API token from environment variable
        self.token = os.environ.get("ESM_FORGE_API_KEY")
        if self.token is None:
            raise ValueError("ESM_FORGE_API_KEY environment variable not set. Please set it to your Forge API token.")
        
        # Initialize clients
        self.client = ESM3InferenceClient(model="esmc-6b-2024-12", url="https://forge.evolutionaryscale.ai", token=self.token)
        self.forge_client = ESM3ForgeInferenceClient(model="esmc-6b-2024-12", url="https://forge.evolutionaryscale.ai", token=self.token)
    
    def run_benchmark(
        self,
        protein: ESMProtein,
        strategy,
        num_trials: int = 5,
        metrics: Optional[List[str]] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run a benchmark for a given protein and denoising strategy.
        
        Args:
            protein: The protein to denoise
            strategy: The denoising strategy to use
            num_trials: Number of trials to run
            metrics: List of metrics to calculate (if None, use all available)
            verbose: Whether to print detailed information
            
        Returns:
            Dict containing benchmark results
        """
        if metrics is None:
            metrics = [
                "uacce",
                "plddt",
                "ptm",
                "foldability",
                "entropy",
                "diversity",
                "cosine_similarity"
            ]
        
        results = {
            "strategy": self.strategy_name,
            "protein_length": len(protein),
            "num_trials": num_trials,
            "metrics": metrics,
            "trials": []
        }
        
        # Store original sequence for comparison
        original_sequence = protein.sequence
        
        # Run trials
        for trial in range(num_trials):
            if verbose:
                print(f"\nRunning trial {trial + 1}/{num_trials}")
            
            trial_start = time.time()
            
            # Run denoising strategy
            denoised_protein = strategy.denoise(protein)
            
            # Calculate metrics
            trial_metrics = {}
            
            if "uacce" in metrics:
                trial_metrics["uacce"] = single_metric_UACCE(self.client, denoised_protein, verbose)
            
            if "plddt" in metrics:
                trial_metrics["plddt"] = single_metric_average_pLDDT(denoised_protein)
            
            if "ptm" in metrics:
                trial_metrics["ptm"] = single_metric_pTM(denoised_protein)
            
            if "foldability" in metrics:
                trial_metrics["foldability"] = single_metric_foldability(self.client, denoised_protein, verbose=verbose)
            
            trial_end = time.time()
            
            # Store trial results
            trial_result = {
                "trial": trial + 1,
                "sequence": denoised_protein.sequence,
                "metrics": trial_metrics,
                "runtime": trial_end - trial_start
            }
            results["trials"].append(trial_result)
            
            if verbose:
                print(f"Trial {trial + 1} completed in {trial_end - trial_start:.2f} seconds")
                print(f"Metrics: {trial_metrics}")
        
        # Calculate aggregated metrics
        sequences = [trial["sequence"] for trial in results["trials"]]
        
        if "entropy" in metrics:
            results["aggregated_metrics"] = {
                "entropy": aggregated_metric_entropy(sequences, verbose=verbose)
            }
        
        if "diversity" in metrics:
            results["aggregated_metrics"]["diversity"] = aggregated_metric_diversity(sequences, verbose=verbose)
        
        if "cosine_similarity" in metrics:
            results["aggregated_metrics"]["cosine_similarity"] = aggregated_cosine_similarities(
                sequences,
                original_sequence,
                verbose=verbose
            )
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """
        Save benchmark results to a JSON file.
        
        Args:
            results: Dictionary containing benchmark results
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.strategy_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}") 