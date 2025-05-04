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

def _run_trial(args):
    """
    Worker function for parallel benchmarking.
    
    Args:
        args: Tuple of (run_id, protein, strategy, metrics, verbose)
        
    Returns:
        dict: Results for this trial
    """
    run_id, protein, strategy, metrics, verbose = args
    
    if verbose:
        print(f"\nRunning trial {run_id + 1}")
    
    trial_start = time.time()
    
    # Run denoising strategy
    denoised_protein = strategy.denoise(protein)
    
    # Calculate metrics
    trial_metrics = {}
    
    if "uacce" in metrics:
        trial_metrics["uacce"] = single_metric_UACCE(strategy.client, denoised_protein, verbose)
    
    if "plddt" in metrics:
        trial_metrics["plddt"] = single_metric_average_pLDDT(denoised_protein)
    
    if "ptm" in metrics:
        trial_metrics["ptm"] = single_metric_pTM(denoised_protein)
    
    if "foldability" in metrics:
        trial_metrics["foldability"] = single_metric_foldability(strategy.client, denoised_protein, verbose=verbose)
    
    trial_end = time.time()
    
    # Store trial results
    trial_result = {
        "trial": run_id + 1,
        "sequence": denoised_protein.sequence,
        "metrics": trial_metrics,
        "runtime": trial_end - trial_start
    }
    
    if verbose:
        print(f"Trial {run_id + 1} completed in {trial_end - trial_start:.2f} seconds")
        print(f"Metrics: {trial_metrics}")
    
    return trial_result

class BenchmarkRunner:
    """
    A class to run benchmarks for protein denoising strategies.
    
    This class handles:
    1. Running multiple trials of a denoising strategy
    2. Collecting metrics for each trial
    3. Aggregating results across trials
    4. Saving results to a JSON file
    """
    
    def __init__(self, client: ESM3InferenceClient, strategy_name: str, output_dir: str = "benchmark_results"):
        """
        Initialize the benchmark runner.
        
        Args:
            client: The ESM3InferenceClient to use for inference
            strategy_name: Name of the denoising strategy being benchmarked
            output_dir: Directory to save benchmark results
        """
        self.client = client
        self.strategy_name = strategy_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
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

    def run_benchmark_parallel(
        self,
        protein: ESMProtein,
        strategy,
        num_trials: int = 5,
        metrics: Optional[List[str]] = None,
        verbose: bool = False,
        n_processes: int = 4
    ) -> Dict[str, Any]:
        """
        Run a benchmark for a given protein and denoising strategy in parallel.
        
        Args:
            protein: The protein to denoise
            strategy: The denoising strategy to use
            num_trials: Number of trials to run
            metrics: List of metrics to calculate (if None, use all available)
            verbose: Whether to print detailed information
            n_processes: Number of parallel processes to use
            
        Returns:
            Dict containing benchmark results
        """
        import multiprocessing as mp
        from functools import partial
        
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
        
        # Create arguments for each worker process
        worker_args = []
        for run_id in range(num_trials):
            worker_args.append((run_id, protein, strategy, metrics, verbose))
        
        # Run trials in parallel
        with mp.Pool(processes=n_processes) as pool:
            trial_results = pool.map(_run_trial, worker_args)
        
        results["trials"] = trial_results
        
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