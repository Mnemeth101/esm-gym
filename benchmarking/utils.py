import os
import json
from typing import List, Dict, Any, Optional
from esm.sdk.api import ESMProtein

def load_protein_from_fasta(fasta_path: str) -> ESMProtein:
    """
    Load a protein sequence from a FASTA file.
    
    Args:
        fasta_path: Path to the FASTA file
        
    Returns:
        ESMProtein object containing the sequence
    """
    with open(fasta_path, "r") as f:
        lines = f.readlines()
    
    # Skip header line and join sequence lines
    sequence = "".join(line.strip() for line in lines[1:])
    
    return ESMProtein(sequence=sequence)

def load_benchmark_results(results_dir: str) -> List[Dict[str, Any]]:
    """
    Load all benchmark results from a directory.
    
    Args:
        results_dir: Directory containing benchmark result JSON files
        
    Returns:
        List of dictionaries containing benchmark results
    """
    results = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "r") as f:
                results.append(json.load(f))
    
    return results

def compare_benchmark_results(
    results: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compare benchmark results across different strategies.
    
    Args:
        results: List of benchmark result dictionaries
        metrics: List of metrics to compare (if None, use all available)
        
    Returns:
        Dictionary containing comparison results
    """
    if metrics is None:
        # Get all metrics from the first result
        metrics = results[0]["metrics"]
    
    comparison = {
        "metrics": metrics,
        "strategies": {}
    }
    
    for result in results:
        strategy = result["strategy"]
        comparison["strategies"][strategy] = {}
        
        # Compare individual trial metrics
        for metric in metrics:
            if metric in ["entropy", "diversity", "cosine_similarity"]:
                # These are aggregated metrics
                if "aggregated_metrics" in result and metric in result["aggregated_metrics"]:
                    comparison["strategies"][strategy][metric] = result["aggregated_metrics"][metric]
            else:
                # These are per-trial metrics
                values = [trial["metrics"].get(metric) for trial in result["trials"]]
                values = [v for v in values if v is not None]
                if values:
                    comparison["strategies"][strategy][metric] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values)
                    }
    
    return comparison

def save_comparison_results(comparison: Dict[str, Any], output_path: str):
    """
    Save benchmark comparison results to a JSON file.
    
    Args:
        comparison: Dictionary containing comparison results
        output_path: Path to save the comparison results
    """
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Comparison results saved to {output_path}") 