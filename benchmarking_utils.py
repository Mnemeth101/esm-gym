from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    LogitsConfig,
    GenerationConfig
)
import torch.nn.functional as F
import math  # Add import for math
from typing import List, Dict, Any, Optional, Tuple, Callable
import time
import os
import sys
from datetime import datetime
from tqdm import tqdm
import inspect
import numpy as np
import json


from denoising_strategies import Tee, BaseDenoisingStrategy

### Single Metrics - metrics for a single protein
def single_metric_UACCE(client: ESM3InferenceClient, protein: ESMProtein, verbose: bool = False) -> float:
    """
    Calculate unmask-aggregated-categorical-cross-entropy (uCCE) for a protein sequence.
    
    This approach directly calculates the log probabilities of observed amino acids
    without using masking or the alpha/beta transformation.
    
    Reference: https://github.com/ntranoslab/esm-variants/issues/12
    
    Args:
        client: Client with inference capability
        protein: Protein sequence to evaluate
        verbose: Whether to print detailed debug information
    
    Returns:
        float: The calculated unmask-categorical-cross-entropy score
    """
    if verbose:
        print(f"Starting uCCE calculation for protein: {protein}")
    
    # 1: Perform inference once to get probabilities
    if verbose:
        print("Step 1: Encoding protein")
    protein_tensor = client.encode(protein)
    if verbose:
        print(f"Encoded protein tensor shape: {protein_tensor.shape if hasattr(protein_tensor, 'shape') else 'N/A'}")
    
    if verbose:
        print("Getting protein tokens")
    protein_tokens = protein_tensor.sequence.numpy()
    if verbose:
        print(f"Protein tokens shape: {protein_tokens.shape}, tokens: {protein_tokens}")
    
    if verbose:
        print("Getting logits from client")
    outputs = client.logits(
        protein_tensor,
        config=LogitsConfig(sequence=True)
    )
    if verbose:
        print(f"Got outputs with attributes: {dir(outputs.logits)}")
    
    logits = outputs.logits.sequence
    if verbose:
        print(f"Logits shape: {logits.shape}")
    
    if verbose:
        print("Calculating probabilities")
    probabilities = F.softmax(logits, dim=-1).cpu().numpy()
    if verbose:
        print(f"Probabilities shape: {probabilities.shape}")
    
    # 2: Extract log probabilities of the actual amino acids
    if verbose:
        print("\nStep 2: Extracting log probabilities")
    log_probs = []
    for i, aa in enumerate(protein_tokens):
        if verbose:
            print(f"Position {i}: token={aa}")
        # Get the probability of the actual amino acid at position i
        p_i = probabilities[i][aa]
        if verbose:
            print(f"  Probability p_i={p_i}")
        
        # Calculate log probability
        log_p = math.log(p_i)
        if verbose:
            print(f"  Log probability: log({p_i}) = {log_p}")
        log_probs.append(log_p)
    
    if verbose:
        print(f"\nLog probabilities (first 5): {log_probs[:5]}")
    
    # 3: Return the mean log probability (excluding first and last tokens)
    if verbose:
        print("\nStep 3: Calculating mean log probability (excluding first and last tokens)")
    
    # Extract only the relevant tokens (skip first and last)
    relevant_log_probs = log_probs[1:-1]
    if verbose:
        print(f"Using log probabilities from position 1 to {len(log_probs)-2}")
        print(f"Number of tokens considered: {len(relevant_log_probs)}")
    
    # Calculate using only the actual protein sequence length
    L = len(protein)
    if verbose:
        print(f"Original protein length: {L}")
    
    sum_log_probs = sum(relevant_log_probs)
    if verbose:
        print(f"Sum of log probabilities (excluding first and last tokens): {sum_log_probs}")
    
    ucce = (1 / L) * sum_log_probs
    if verbose:
        print(f"Final uCCE score: {ucce}")
    
    return ucce

def single_metric_average_pLDDT(protein: ESMProtein) -> float:
    """
    Calculate pLDDT score for a protein sequence averaged over all residues.
    
    Args:
        protein: Protein sequence to evaluate
    
    Returns:
        float: The calculated pLDDT score
    """
    plddts = protein.plddt
    if plddts is None:
        raise ValueError("pLDDT scores are not available for this protein.")

    plddt = plddts.mean()
    return plddt

def single_metric_pTM(protein: ESMProtein) -> float:
    """
    Return the pTM score for a protein structure.
    """
    pTM = protein.ptm
    if pTM is None:
        raise ValueError("pTM scores are not available for this protein.")
    
    return pTM

def single_metric_foldability(client: ESM3InferenceClient, protein: ESMProtein, num_samples: int = 5, verbose: bool = False) -> float:
    """
    Foldability is a measure of how well a protein sequence can fold into a stable structure.
    
    We define foldability as the percentage of sequences satisfying:
    - pLDDT > 80 (per-residue confidence)
    - pTM > 0.7 (template modeling score)
    - pAE < 10 (predicted aligned error - not directly measured here)
    
    This method uses the existing sequence from the input protein and tests these criteria.
    
    Args:
        client: Client with inference capability
        protein: Protein with sequence to evaluate
        num_samples: Number of samples to test (default: 5)
        verbose: Whether to print detailed information during execution
        
    Returns:
        float: The percentage of samples that satisfy the foldability criteria
    """
    # Validate that the protein has a sequence
    if protein.sequence is None:
        raise ValueError("Protein sequence is not available for this protein.")
    
    def just_keep_sequence(protein: ESMProtein) -> ESMProtein:
        protein_copy = protein.copy()
        protein_copy.structure = None
        protein_copy.plddt = None
        protein_copy.ptm = None
        protein_copy.secondary_structure = None
        protein_copy.sasa = None
        protein_copy.functional_annotations = None
        return protein_copy

    original_protein = protein.copy()
    successful_count = 0
    
    if verbose:
        print(f"Testing foldability with {num_samples} samples")
        print(f"Using sequence: {original_protein.sequence}")
        
    for i in range(num_samples):
        if verbose:
            print(f"Sample {i+1}/{num_samples}")
            
        try:
            # Start with just the sequence from the original protein
            protein = just_keep_sequence(original_protein)
            
            # Predict the structure of the sequence
            protein = client.generate(
                input=protein,
                config=GenerationConfig(track="structure", num_steps=1),
            )
            if isinstance(protein, ESMProteinError):
                raise ValueError("Protein structure generation failed.")
            
            # Check foldability criteria
            plddt = protein.plddt.mean()
            ptm = protein.ptm
            
            if verbose:
                print(f"  pLDDT: {plddt:.2f}, pTM: {ptm:.2f}")
                
            # Check if this sample meets all criteria
            if plddt > .80 and ptm > 0.7:
                successful_count += 1
                if verbose:
                    print("  ✓ Sample meets foldability criteria")
            else:
                if verbose:
                    print("  ✗ Sample does not meet foldability criteria")
                    
        except ValueError as e:
            if verbose:
                print(f"  Error in sample {i+1}: {str(e)}")
    
    # Calculate foldability as percentage of successful samples
    foldability = successful_count / num_samples if num_samples > 0 else 0.0
    
    if verbose:
        print(f"Foldability score: {foldability * 100:.1f}% ({successful_count}/{num_samples} samples passed)")
        
    return foldability


### Aggregated Metrics - metrics over all proteins generated using a denoising strategy
def aggregated_metric_entropy(proteins: List[str], position_specific=True, verbose=False):
    """
    Calculate the entropy of protein sequences generated over N runs of a denoising strategy.
    
    Args:
        proteins: List of protein sequences to evaluate (as strings)
        position_specific: If True, calculate position-specific entropy; otherwise global
        verbose: Whether to print detailed information
        
    Returns:
        float or list: Position-specific entropy (list) or global entropy (float)
    """
    import numpy as np
    from collections import Counter
    
    if not proteins or len(proteins) == 0:
        raise ValueError("No protein sequences provided")
        
    # Define the standard amino acid alphabet
    amino_acids = "ACDEFGHIKLMNPQRSTVWYXZ"
    
    if position_specific:
        # Make sure all sequences are the same length
        seq_len = len(proteins[0])
        if not all(len(seq) == seq_len for seq in proteins):
            raise ValueError("All protein sequences must be the same length for position-specific entropy")
            
        # Calculate position-specific entropy
        entropies = []
        
        for pos in range(seq_len):
            # Count amino acids at this position
            pos_aas = [seq[pos] for seq in proteins]
            counts = Counter(pos_aas)
            
            # Calculate entropy at this position
            total_count = len(proteins)
            pos_entropy = 0
            
            for aa in amino_acids:
                prob = counts.get(aa, 0) / total_count
                if prob > 0:
                    pos_entropy -= prob * math.log2(prob)
                    
            entropies.append(pos_entropy)
            
            if verbose and pos < 5:  # Show first few positions
                print(f"Position {pos}: entropy = {pos_entropy:.4f}, distribution: {counts}")
                
        if verbose:
            print(f"Average position-specific entropy: {np.mean(entropies):.4f}")
        
        return entropies
    else:
        # Calculate global amino acid distribution entropy
        all_aas = "".join(proteins)
        counts = Counter(all_aas)
        
        # Calculate entropy
        total_count = len(all_aas)
        entropy = 0
        
        for aa in amino_acids:
            prob = counts.get(aa, 0) / total_count
            if prob > 0:
                entropy -= prob * math.log2(prob)
                
        if verbose:
            print(f"Global entropy: {entropy:.4f}")
            print(f"Amino acid distribution: {counts}")
            
        return entropy

def levenshtein_distance(seq1, seq2):
    """
    Calculate the Levenshtein (edit) distance between two sequences.
    
    Args:
        seq1, seq2: The two sequences to compare
        
    Returns:
        int: The edit distance
    """
    # Create a matrix of size (len(seq1)+1) x (len(seq2)+1)
    rows, cols = len(seq1) + 1, len(seq2) + 1
    dist = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Initialize the first row and column
    for i in range(rows):
        dist[i][0] = i
    for j in range(cols):
        dist[0][j] = j
        
    # Fill the matrix
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dist[i][j] = min(
                dist[i-1][j] + 1,      # deletion
                dist[i][j-1] + 1,      # insertion
                dist[i-1][j-1] + cost  # substitution
            )
            
    return dist[rows-1][cols-1]

def aggregated_metric_diversity(proteins: List[str], method="levenshtein", verbose=False):
    """
    Calculate the diversity of protein sequences generated over N runs of a denoising strategy.
    
    Args:
        proteins: List of protein sequences to evaluate (as strings)
        method: Method to calculate diversity:
                "levenshtein" - average normalized Levenshtein distance
                "hamming" - average normalized Hamming distance (sequences must be same length)
                "identity" - average pairwise identity percentage (1 - identity)
        verbose: Whether to print detailed information
        
    Returns:
        float: Diversity score between 0 (identical) and 1 (maximum diversity)
    """
    import numpy as np
    
    if not proteins or len(proteins) < 2:
        raise ValueError("Need at least 2 protein sequences to calculate diversity")
    
    num_proteins = len(proteins)
    distances = []
    
    for i in range(num_proteins):
        for j in range(i+1, num_proteins):
            seq1 = proteins[i]
            seq2 = proteins[j]
            
            if method == "levenshtein":
                # Levenshtein (edit) distance
                distance = levenshtein_distance(seq1, seq2)
                # Normalize by the length of the longer sequence
                normalized_distance = distance / max(len(seq1), len(seq2))
                
            elif method == "hamming":
                # Hamming distance (sequences must be same length)
                if len(seq1) != len(seq2):
                    raise ValueError("All sequences must be the same length for Hamming distance")
                distance = sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
                normalized_distance = distance / len(seq1)
                
            elif method == "identity":
                # Sequence identity percentage
                if len(seq1) != len(seq2):
                    raise ValueError("All sequences must be the same length for identity calculation")
                identity = sum(c1 == c2 for c1, c2 in zip(seq1, seq2)) / len(seq1)
                normalized_distance = 1 - identity
                
            else:
                raise ValueError(f"Unknown diversity method: {method}")
                
            distances.append(normalized_distance)
            
            if verbose and len(distances) <= 3:  # Show first few comparisons
                print(f"Distance between sequence {i} and {j}: {normalized_distance:.4f}")
    
    # Calculate average distance
    avg_distance = np.mean(distances)
    
    if verbose:
        print(f"Average {method} distance: {avg_distance:.4f} (from {len(distances)} comparisons)")
        print(f"Min distance: {np.min(distances):.4f}, Max distance: {np.max(distances):.4f}")
        
    return avg_distance

def aggregated_cosine_similarities(proteins: List[str], original_protein: str, verbose=False):
    """
    Calculate cosine similarities between embeddings of generated proteins and an original protein.
    
    Args:
        proteins: List of protein sequences to evaluate (as strings)
        original_protein: The original protein sequence to compare against
        verbose: Whether to print detailed information
        
    Returns:
        List[float]: Cosine similarities between each protein and the original protein
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from esm.sdk.forge import ESM3ForgeInferenceClient
    from esm.sdk.api import ESMProtein, LogitsConfig
    import os
    
    # Get API token from environment variable or prompt user
    token = os.environ.get("ESM_FORGE_API_KEY")
    if token is None:
        raise ValueError("ESM_FORGE_API_KEY environment variable not set. Please set it to your Forge API token.")
    
    # Initialize client
    if verbose:
        print("Initializing ESM-C forge client...")
    forge_client = ESM3ForgeInferenceClient(model="esmc-6b-2024-12", url="https://forge.evolutionaryscale.ai", token=token)
    
    # Generate embedding for the original protein
    if verbose:
        print(f"Generating embedding for original protein (length: {len(original_protein)})")
    
    original_protein_obj = ESMProtein(sequence=original_protein)
    original_tensor = forge_client.encode(original_protein_obj)
    original_output = forge_client.logits(original_tensor, config=LogitsConfig(return_embeddings=True))
    
    # Get the embedding vector (average across all layers for robust representation)
    original_embedding = original_output.embeddings.mean(dim=1).squeeze().detach()  # Shape: [2560]
    
    # Normalize the embedding
    original_embedding_norm = F.normalize(original_embedding, p=2, dim=0)
    
    similarities = []
    
    # Calculate similarities for each protein
    for i, protein_seq in enumerate(proteins):
        if verbose and (i == 0 or i % 5 == 0):
            print(f"Processing protein {i+1}/{len(proteins)} (length: {len(protein_seq)})")
            
        try:
            # Generate embedding for this protein
            protein_obj = ESMProtein(sequence=protein_seq)
            protein_tensor = forge_client.encode(protein_obj)
            protein_output = forge_client.logits(protein_tensor, config=LogitsConfig(return_embeddings=True))
            
            # Get the embedding vector (average across all layers)
            protein_embedding = protein_output.embeddings.mean(dim=1).squeeze().detach()  # Shape: [2560]
            
            # Normalize the embedding
            protein_embedding_norm = F.normalize(protein_embedding, p=2, dim=0)
            
            # Calculate cosine similarity (dot product of normalized vectors)
            similarity = torch.dot(original_embedding_norm, protein_embedding_norm).item()
            similarities.append(similarity)
            
            if verbose and i < 3:  # Show first few similarities
                print(f"  Similarity between original and protein {i+1}: {similarity:.4f}")
                
        except Exception as e:
            if verbose:
                print(f"Error processing protein {i+1}: {e}")
            similarities.append(None)
    
    if verbose:
        valid_similarities = [s for s in similarities if s is not None]
        if valid_similarities:
            print(f"Average similarity: {np.mean(valid_similarities):.4f}")
            print(f"Min similarity: {np.min(valid_similarities):.4f}")
            print(f"Max similarity: {np.max(valid_similarities):.4f}")
    
    return similarities

### Benchmarking run utils
def get_smallest_pdb_file():
    """
    Find the PDB file with the fewest number of residues in ./data/casp15_monomers_without_T1137
    Turns out to be T1119-D1.pdb!
    
    Returns:
        str: Path to the PDB file with fewest residues
    """
    import os
    import glob
    from Bio.PDB import PDBParser
    
    # Define the directory path
    pdb_dir = "./data/casp15_monomers_without_T1137"
    
    # Get all PDB files in the directory
    pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
    
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in {pdb_dir}")
    
    # Initialize PDB parser
    parser = PDBParser(QUIET=True)
    
    smallest_file = None
    min_residues = float('inf')
    
    # Iterate through each PDB file and count residues
    for pdb_file in pdb_files:
        try:
            # Get the structure ID from filename
            structure_id = os.path.basename(pdb_file).split('.')[0]
            
            # Parse the structure
            structure = parser.get_structure(structure_id, pdb_file)
            
            # Count residues (exclude water and hetero residues)
            residue_count = sum(1 for residue in structure.get_residues() 
                               if residue.id[0] == ' ')  # Standard residues have ' ' as hetflag
            
            print(f"File: {pdb_file}, Residue count: {residue_count}")
            # Update if this is the smallest so far
            if residue_count < min_residues:
                min_residues = residue_count
                smallest_file = pdb_file
                
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
            continue
    
    if smallest_file is None:
        raise ValueError("Could not determine the PDB file with fewest residues")
    
    print(f"PDB file with fewest residues: {smallest_file} ({min_residues} residues)")
    
    return smallest_file

# Define a worker function that will be used by run_benchmark_parallel
# This needs to be at module level to be picklable
def _parallel_benchmark_worker(args):
    """
    Worker function for parallel benchmarking.
    
    Args:
        args: Tuple of (run_id, strategy_params, source_protein, strategy_class_name, verbose)
        
    Returns:
        dict: Results for this run
    """
    import time
    import io
    import copy
    import importlib
    
    run_id, strategy_params, source_protein, strategy_class_name, verbose = args
    
    # Create a copy of the strategy for this worker
    # We need to dynamically import the strategy class from the denoising_strategies module
    try:
        denoising_module = importlib.import_module('denoising_strategies')
        strategy_class = getattr(denoising_module, strategy_class_name)
        worker_strategy = strategy_class(**strategy_params)
    except Exception as e:
        return {
            "run_id": run_id,
            "success": False,
            "error": f"Failed to create strategy: {str(e)}",
            "time": 0,
            "log": f"Error creating strategy: {str(e)}\n"
        }
    
    # Capture output for this worker in a string buffer
    worker_output = io.StringIO()
    worker_output.write(f"--- Run {run_id+1} ---\n")
    
    # Initialize result data
    result = {
        "run_id": run_id,
        "success": False,
        "protein": None,
        "metrics": {},
        "cost": 0,
        "time": 0,
        "error": None,
        "log": ""
    }
    
    start_time = time.time()
    
    try:
        # Reset the strategy instance if it has reset method
        if hasattr(worker_strategy, 'reset') and callable(worker_strategy.reset):
            worker_strategy.reset()
        
        # Run the denoising strategy
        generated_protein = worker_strategy.denoise(
            source_protein.copy(),
            verbose=False  # Less verbose in parallel mode
        )
        
        # Get the protein and cost from the strategy
        if hasattr(worker_strategy, 'return_generation'):
            protein_and_cost = worker_strategy.return_generation()
            generated_protein = protein_and_cost[0] if generated_protein is None else generated_protein
            cost = protein_and_cost[1] if len(protein_and_cost) > 1 else 0
        else:
            cost = getattr(worker_strategy, 'cost', 0)
        
        end_time = time.time()
        run_time = end_time - start_time
        
        # Check if we got a valid protein
        if generated_protein is not None and generated_protein.sequence:
            # Store important protein attributes in a serializable format
            protein_data = {
                "sequence": generated_protein.sequence,
            }
            
            # Add structure if available (simplified)
            if hasattr(generated_protein, 'structure') and generated_protein.structure is not None:
                protein_data["has_structure"] = True
            
            # Add pLDDT and pTM if available
            if hasattr(generated_protein, 'plddt') and generated_protein.plddt is not None:
                protein_data["plddt"] = generated_protein.plddt.tolist()
            if hasattr(generated_protein, 'ptm') and generated_protein.ptm is not None:
                protein_data["ptm"] = float(generated_protein.ptm)
            
            # Collect metrics but don't compute UACCE or foldability in workers
            worker_metrics = {}
            if hasattr(generated_protein, 'plddt') and generated_protein.plddt is not None:
                try:
                    from benchmarking_utils import single_metric_average_pLDDT
                    worker_metrics['avg_pLDDT'] = single_metric_average_pLDDT(generated_protein)
                except Exception as e:
                    worker_output.write(f"Error calculating pLDDT: {e}\n")
            
            if hasattr(generated_protein, 'ptm') and generated_protein.ptm is not None:
                try:
                    from benchmarking_utils import single_metric_pTM
                    worker_metrics['pTM'] = single_metric_pTM(generated_protein)
                except Exception as e:
                    worker_output.write(f"Error calculating pTM: {e}\n")
            
            result["success"] = True
            result["protein_data"] = protein_data
            result["generated_protein"] = generated_protein  # Keep the full protein object for post-processing
            result["metrics"] = worker_metrics
            result["cost"] = cost
            result["time"] = run_time
            
            worker_output.write(f"Run {run_id+1} completed in {run_time:.2f}s with cost {cost}\n")
            worker_output.write(f"Generated sequence: {generated_protein.sequence}\n")
            worker_output.write(f"Metrics: {worker_metrics}\n")
        else:
            result["error"] = "No sequence generated"
            worker_output.write(f"Run {run_id+1} failed: no sequence generated\n")
    
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        worker_output.write(f"Error in run {run_id+1}: {e}\n")
        worker_output.write(error_msg + "\n")
        result["error"] = str(e)
        result["time"] = time.time() - start_time
    
    # Save the captured output
    result["log"] = worker_output.getvalue()
    worker_output.close()
    
    return result

class BenchmarkRunner:
    """
    Simplified class to benchmark denoising strategies by running them multiple times and collecting metrics.
    
    This version is more flexible and can work with any denoising strategy regardless of its specific parameters.
    It captures individual metrics for each protein and aggregates them at the end.
    """
    
    def __init__(
        self, 
        client,
        source_protein_path,
        num_runs=50,
        verbose=False,
        output_dir="data/benchmarking_results"
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            client: ESM3InferenceClient instance
            source_protein_path: Path to the source protein PDB file
            num_runs: Number of runs (generated sequences) per strategy
            verbose: Whether to print detailed information
            output_dir: Base directory for benchmark results
        """
        self.client = client
        self.source_protein_path = source_protein_path
        self.num_runs = num_runs
        self.verbose = verbose
        self.results = {}
        self.output_dir = output_dir
        # Load the source protein from the provided path
        self.source_protein = ESMProtein.from_pdb(source_protein_path)
        if self.verbose:
            print(f"Loaded source protein from {source_protein_path}")
            print(f"Source protein sequence length: {len(self.source_protein.sequence)}")
    
    def _create_job_folder(self, strategy_name):
        """
        Create a job folder for this benchmark run.
        
        Args:
            strategy_name: Name of the strategy being benchmarked
        
        Returns:
            tuple: (job_folder_path, log_file_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"{strategy_name}_{timestamp}"
        job_folder = os.path.join(self.output_dir, job_name)
        
        # Ensure the directory exists
        os.makedirs(job_folder, exist_ok=True)
        
        # Create a log file path
        log_file = os.path.join(job_folder, "denoising_output.txt")
        
        print(f"Created job folder: {job_folder}")
        return job_folder, log_file
    
    def _save_sequences(self, job_folder, generated_proteins, strategy_name):
        """
        Save the generated sequences to a file.
        
        Args:
            job_folder: Path to the job folder
            generated_proteins: List of generated proteins
            strategy_name: Name of the strategy
            
        Returns:
            str: Path to the saved file
        """
        sequences_file = os.path.join(job_folder, "generated_sequences.txt")
        with open(sequences_file, 'w') as f:
            f.write(f"Generated sequences for {strategy_name}\n")
            f.write(f"Total sequences: {len(generated_proteins)}\n\n")
            
            for i, protein in enumerate(generated_proteins):
                f.write(f">Sequence_{i+1}\n")
                f.write(f"{protein.sequence}\n\n")
                
                # Add additional protein metadata if available
                if hasattr(protein, 'plddt') and protein.plddt is not None:
                    f.write(f"Average pLDDT: {protein.plddt.mean():.4f}\n")
                if hasattr(protein, 'ptm') and protein.ptm is not None:
                    f.write(f"pTM: {protein.ptm:.4f}\n")
                f.write("\n")
                
        print(f"Saved {len(generated_proteins)} sequences to {sequences_file}")
        return sequences_file
    
    def _save_stats(self, job_folder, results, strategy_name):
        """
        Save the benchmark statistics to a file.
        
        Args:
            job_folder: Path to the job folder
            results: Benchmark results
            strategy_name: Name of the strategy
            
        Returns:
            str: Path to the saved file
        """
        stats_file = os.path.join(job_folder, "benchmark_stats.json")
        
        # Convert results to serializable format
        serializable_results = {}
        
        # Make a copy to avoid modifying the original
        serialized_result = {}
        
        # Selectively copy fields that can be serialized
        for key, value in results.items():
            # Skip non-serializable objects
            if key in ['generated_proteins', 'single_metrics']:
                continue
                
            if key == 'strategy_params':
                # Create a copy of strategy params without the client
                serialized_result[key] = {k: v for k, v in value.items() if k != 'client'}
                continue
                
            if key == 'avg_single_metrics':
                # Convert any numpy values to regular Python types
                serialized_result[key] = {
                    metric: val.item() if hasattr(val, 'item') else val
                    for metric, val in value.items()
                }
                continue
                
            # Handle numpy values
            if hasattr(value, 'item'):
                serialized_result[key] = value.item()
            elif isinstance(value, dict):
                # Handle nested dictionaries
                serialized_result[key] = {}
                for k, v in value.items():
                    if hasattr(v, 'item'):
                        serialized_result[key][k] = v.item()
                    else:
                        serialized_result[key][k] = v
            elif isinstance(value, (list, np.ndarray)) and key in ["costs", "times"]:
                # Convert costs and times to lists
                serialized_result[key] = [x.item() if hasattr(x, 'item') else x for x in value]
            else:
                # Other simple types
                serialized_result[key] = value
                
        # Record counts instead of full sequences
        if 'generated_proteins' in results:
            serialized_result['num_generated'] = len(results['generated_proteins'])
                
        serializable_results[strategy_name] = serialized_result
        
        # Save to file
        with open(stats_file, 'w') as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "source_protein": {
                        "path": self.source_protein_path,
                        "sequence": self.source_protein.sequence,
                        "length": len(self.source_protein.sequence)
                    },
                    "config": {
                        "num_runs": self.num_runs
                    },
                    "results": serializable_results
                },
                f,
                indent=2
            )
            
        print(f"Saved benchmark stats to {stats_file}")
        return stats_file
    
    def run_benchmark(self, strategy_instance: BaseDenoisingStrategy, strategy_name=None):
        """
        Run a benchmark for a specific denoising strategy.
        
        Args:
            strategy_instance: Instance of a denoising strategy (must have denoise method and return_generation method)
            strategy_name: Name for this strategy in results (defaults to class name)
            
        Returns:
            dict: Benchmark results
        """
        if strategy_name is None:
            strategy_name = strategy_instance.__class__.__name__
            
        print(f"\n{'=' * 80}\nRunning benchmark for: {strategy_name}\n{'=' * 80}")
        
        # Create job folder and log file
        job_folder, log_file = self._create_job_folder(strategy_name)
        
        # Create output file and backup stdout
        original_stdout = sys.stdout
        sys.stdout = Tee(log_file)
        
        # Extract strategy parameters for logging
        strategy_params = strategy_instance._extract_strategy_params() # Require this from the Denoisers classes
        
        print(f"Benchmark started at {datetime.now()}")
        print(f"Strategy: {strategy_name}")
        print(f"Strategy parameters: {strategy_params}")
        print(f"Source protein: {self.source_protein.sequence}")
        print(f"Number of runs: {self.num_runs}")
        
        # Lists to collect results
        generated_proteins = []
        single_metrics = []  # Will contain dictionaries of metrics for each protein
        costs = []
        times = []
        
        try:
            for run in tqdm(range(self.num_runs), desc=f"Benchmarking {strategy_name}"):
                print(f"\n\n--- Run {run+1}/{self.num_runs} ---")
                
                # Reset the strategy instance if it has reset method
                if hasattr(strategy_instance, 'reset') and callable(strategy_instance.reset):
                    strategy_instance.reset()
                
                start_time = time.time()
                
                # Run the denoising strategy
                try:
                    generated_protein = strategy_instance.denoise(
                        self.source_protein.copy(), 
                        verbose=self.verbose
                    )
                    
                    # Get the protein and cost from the strategy
                    if hasattr(strategy_instance, 'return_generation'):
                        protein_and_cost = strategy_instance.return_generation()
                        generated_protein = protein_and_cost[0] if generated_protein is None else generated_protein
                        cost = protein_and_cost[1] if len(protein_and_cost) > 1 else 0
                    else:
                        cost = getattr(strategy_instance, 'cost', 0)
                    
                    end_time = time.time()
                    run_time = end_time - start_time
                    
                    # Collect metrics for this individual protein
                    protein_metrics = self._collect_single_metrics(generated_protein)
                    
                    # Store results
                    if generated_protein.sequence:
                        generated_proteins.append(generated_protein)
                        single_metrics.append(protein_metrics)
                        times.append(run_time)
                        costs.append(cost)
                        
                        print(f"Run {run+1} completed in {run_time:.2f}s with cost {cost}")
                        print(f"Generated sequence: {generated_protein.sequence}")
                        print(f"Single metrics: {protein_metrics}")
                    else:
                        print(f"Run {run+1} failed: no sequence generated")
                        
                except Exception as e:
                    print(f"Error in run {run+1}: {e}")
                    import traceback
                    traceback.print_exc()
                
            # Calculate aggregated metrics
            print("\n\n--- Benchmark Results ---")
            
            # Only calculate metrics if we have generated proteins
            if generated_proteins:
                sequences = [p.sequence for p in generated_proteins]
                
                # Calculate diversity using different methods
                try:
                    levenshtein_diversity = aggregated_metric_diversity(
                        sequences, method="levenshtein", verbose=True
                    )
                    if all(len(p) == len(sequences[0]) for p in sequences):
                        hamming_diversity = aggregated_metric_diversity(
                            sequences, method="hamming", verbose=True
                        )
                        identity_diversity = aggregated_metric_diversity(
                            sequences, method="identity", verbose=True
                        )
                    else:
                        hamming_diversity = None
                        identity_diversity = None
                except Exception as e:
                    print(f"Error calculating diversity metrics: {e}")
                    levenshtein_diversity = None
                    hamming_diversity = None
                    identity_diversity = None
                    
                # Calculate entropy
                try:
                    global_entropy = aggregated_metric_entropy(
                        sequences, position_specific=False, verbose=True
                    )
                    if all(len(p) == len(sequences[0]) for p in sequences):
                        position_entropies = aggregated_metric_entropy(
                            sequences, position_specific=True, verbose=True
                        )
                        avg_position_entropy = sum(position_entropies) / len(position_entropies) if position_entropies else None
                    else:
                        position_entropies = None
                        avg_position_entropy = None
                except Exception as e:
                    print(f"Error calculating entropy metrics: {e}")
                    global_entropy = None
                    position_entropies = None
                    avg_position_entropy = None
                
                # Calculate statistics
                avg_cost = np.mean(costs) if costs else 0
                std_cost = np.std(costs) if costs else 0
                avg_time = np.mean(times) if times else 0
                std_time = np.std(times) if times else 0
                
                # Average single metrics across all proteins
                avg_single_metrics = self._average_single_metrics(single_metrics)
                
                # Compile results
                results = {
                    "strategy": strategy_name,
                    "strategy_params": strategy_params,
                    "num_runs": self.num_runs,
                    "num_completed": len(generated_proteins),
                    "avg_cost": avg_cost,
                    "std_cost": std_cost,
                    "avg_time": avg_time,
                    "std_time": std_time,
                    "entropy": {
                        "global": global_entropy,
                        "avg_position": avg_position_entropy
                    },
                    "diversity": {
                        "levenshtein": levenshtein_diversity,
                        "hamming": hamming_diversity,
                        "identity": identity_diversity
                    },
                    "avg_single_metrics": avg_single_metrics,
                    "single_metrics": single_metrics,  # Store all individual metrics
                    "generated_proteins": generated_proteins,
                    "costs": costs,
                    "times": times
                }
                
                # Add the source protein name or path to the results
                source_protein_name = getattr(self.source_protein, "name", None)
                if not source_protein_name:
                    source_protein_name = self.source_protein_path or "unknown_protein"
                results["source_protein_name"] = source_protein_name
                
                # Print summary
                print("\n--- Summary ---")
                print(f"Strategy: {strategy_name}")
                print(f"Strategy parameters: {strategy_params}")
                print(f"Completed runs: {len(generated_proteins)}/{self.num_runs}")
                print(f"Average cost: {avg_cost:.2f} ± {std_cost:.2f}")
                print(f"Average time: {avg_time:.2f}s ± {std_time:.2f}s")
                print(f"Global entropy: {global_entropy:.4f}" if global_entropy else "Global entropy: N/A")
                print(f"Average position-specific entropy: {avg_position_entropy:.4f}" if avg_position_entropy else "Average position entropy: N/A")
                print(f"Levenshtein diversity: {levenshtein_diversity:.4f}" if levenshtein_diversity else "Levenshtein diversity: N/A")
                print(f"Average metrics across proteins:")
                for metric, value in avg_single_metrics.items():
                    print(f"  - {metric}: {value:.4f}")
                
                # Store results
                self.results[strategy_name] = results
                
                # Save sequences and stats to job folder
                self._save_sequences(job_folder, generated_proteins, strategy_name)
                self._save_stats(job_folder, results, strategy_name)
                
                return results
                
            else:
                print("No proteins were successfully generated.")
                results = {
                    "strategy": strategy_name,
                    "strategy_params": strategy_params,
                    "num_runs": self.num_runs,
                    "num_completed": 0,
                    "error": "No proteins were successfully generated."
                }
                
                # Save stats even if no proteins were generated
                self._save_stats(job_folder, results, strategy_name)
                return results
                
        except Exception as e:
            print(f"Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            
            results = {
                "strategy": strategy_name,
                "strategy_params": strategy_params,
                "error": str(e),
                "num_completed": len(generated_proteins) if 'generated_proteins' in locals() else 0
            }
            
            # Save stats even on error
            if 'job_folder' in locals():
                self._save_stats(job_folder, results, strategy_name)
                
                # Save sequences if we have any
                if 'generated_proteins' in locals() and generated_proteins:
                    self._save_sequences(job_folder, generated_proteins, strategy_name)
            
            return results
            
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            print(f"Benchmark for {strategy_name} completed. Log saved to {log_file}")
    
    def run_benchmark_parallel(self, strategy_instance: BaseDenoisingStrategy, strategy_name=None, n_processes=None):
        """
        Run a benchmark for a specific denoising strategy using parallel processing.
        
        Args:
            strategy_instance: Instance of a denoising strategy (must have denoise method and return_generation method)
            strategy_name: Name for this strategy in results (defaults to class name)
            n_processes: Number of processes to use (defaults to number of CPU cores)
            
        Returns:
            dict: Benchmark results
        """
        import multiprocessing as mp
        from multiprocessing import Pool
        import copy
        from functools import partial
        
        if strategy_name is None:
            strategy_name = strategy_instance.__class__.__name__
            
        print(f"\n{'=' * 80}\nRunning parallel benchmark for: {strategy_name}\n{'=' * 80}")
        
        # Create job folder and log file
        job_folder, log_file = self._create_job_folder(strategy_name)
        
        # Determine number of processes
        if n_processes is None:
            n_processes = mp.cpu_count()
        n_processes = min(n_processes, self.num_runs)  # Don't use more processes than runs
        
        print(f"Using {n_processes} parallel processes for {self.num_runs} runs")
        
        # Extract strategy parameters for logging and add client
        strategy_params = strategy_instance._extract_strategy_params()
        strategy_params['client'] = self.client  # Add client to parameters
        
        # Main file for final aggregated results
        with open(log_file, 'w') as main_log:
            main_log.write(f"Parallel benchmark started at {datetime.now()}\n")
            main_log.write(f"Strategy: {strategy_name}\n")
            main_log.write(f"Strategy parameters: {strategy_params}\n")
            main_log.write(f"Source protein: {self.source_protein.sequence}\n")
            main_log.write(f"Number of runs: {self.num_runs}\n")
            main_log.write(f"Number of processes: {n_processes}\n\n")
        
        # Create arguments for each worker process
        worker_args = []
        for run_id in range(self.num_runs):
            # Only pass serializable data to the worker
            worker_args.append((
                run_id,
                strategy_params,
                self.source_protein,
                strategy_instance.__class__.__name__,
                self.verbose
            ))
        
        # Execute the worker function in parallel
        try:
            # Create pool and run tasks
            with Pool(processes=n_processes) as pool:
                # Use imap_unordered to get results as they complete
                all_results = list(tqdm(
                    pool.imap_unordered(_parallel_benchmark_worker, worker_args),
                    total=self.num_runs,
                    desc=f"Parallel benchmark for {strategy_name}"
                ))
            
            # Process and aggregate results
            generated_proteins = []
            single_metrics = []
            costs = []
            times = []
            
            # First, write all logs in order of run_id
            with open(log_file, 'a') as main_log:
                # Sort results by run_id for ordered logging
                sorted_results = sorted(all_results, key=lambda x: x["run_id"])
                for result in sorted_results:
                    main_log.write(f"\n{result['log']}\n")
                    
                    # Collect successful runs
                    if result["success"]:
                        # Here we're dealing with reconstructed proteins that might not have all methods
                        # so we'll just collect what we have
                        if "generated_protein" in result:
                            generated_proteins.append(result["generated_protein"])
                            single_metrics.append(result["metrics"])
                        costs.append(result["cost"])
                        times.append(result["time"])
            
            # Calculate metrics that require the original client
            # First, calculate individual protein metrics that couldn't be done in workers
            print(f"Calculating additional metrics for {len(generated_proteins)} proteins...")
            for i, protein in enumerate(generated_proteins):
                if hasattr(self, 'client') and self.client is not None:
                    try:
                        # Add UACCE and possibly other metrics requiring the client
                        single_metrics[i]['UACCE'] = single_metric_UACCE(self.client, protein, verbose=False)
                    except Exception as e:
                        print(f"Error calculating UACCE for protein {i}: {e}")
            
            # Now calculate aggregated metrics if we have generated proteins
            if generated_proteins:
                sequences = [p.sequence for p in generated_proteins]
                
                # Calculate diversity using different methods
                try:
                    levenshtein_diversity = aggregated_metric_diversity(
                        sequences, method="levenshtein", verbose=True
                    )
                    if all(len(p) == len(sequences[0]) for p in sequences):
                        hamming_diversity = aggregated_metric_diversity(
                            sequences, method="hamming", verbose=True
                        )
                        identity_diversity = aggregated_metric_diversity(
                            sequences, method="identity", verbose=True
                        )
                    else:
                        hamming_diversity = None
                        identity_diversity = None
                except Exception as e:
                    print(f"Error calculating diversity metrics: {e}")
                    levenshtein_diversity = None
                    hamming_diversity = None
                    identity_diversity = None
                    
                # Calculate entropy
                try:
                    global_entropy = aggregated_metric_entropy(
                        sequences, position_specific=False, verbose=True
                    )
                    if all(len(p) == len(sequences[0]) for p in sequences):
                        position_entropies = aggregated_metric_entropy(
                            sequences, position_specific=True, verbose=True
                        )
                        avg_position_entropy = sum(position_entropies) / len(position_entropies) if position_entropies else None
                    else:
                        position_entropies = None
                        avg_position_entropy = None
                except Exception as e:
                    print(f"Error calculating entropy metrics: {e}")
                    global_entropy = None
                    position_entropies = None
                    avg_position_entropy = None
                
                # Calculate statistics
                avg_cost = np.mean(costs) if costs else 0
                std_cost = np.std(costs) if costs else 0
                avg_time = np.mean(times) if times else 0
                std_time = np.std(times) if times else 0
                
                # Average single metrics across all proteins
                avg_single_metrics = self._average_single_metrics(single_metrics)
                
                # Compile results
                results = {
                    "strategy": strategy_name,
                    "strategy_params": strategy_params,
                    "num_runs": self.num_runs,
                    "num_completed": len(generated_proteins),
                    "avg_cost": avg_cost,
                    "std_cost": std_cost,
                    "avg_time": avg_time,
                    "std_time": std_time,
                    "entropy": {
                        "global": global_entropy,
                        "avg_position": avg_position_entropy
                    },
                    "diversity": {
                        "levenshtein": levenshtein_diversity,
                        "hamming": hamming_diversity,
                        "identity": identity_diversity
                    },
                    "avg_single_metrics": avg_single_metrics,
                    "single_metrics": single_metrics,  # Store all individual metrics
                    "generated_proteins": generated_proteins,
                    "costs": costs,
                    "times": times
                }
                
                # Add the source protein name or path to the results
                source_protein_name = getattr(self.source_protein, "name", None)
                if not source_protein_name:
                    source_protein_name = self.source_protein_path or "unknown_protein"
                results["source_protein_name"] = source_protein_name
                
                # Print summary
                print("\n--- Summary ---")
                print(f"Strategy: {strategy_name}")
                print(f"Strategy parameters: {strategy_params}")
                print(f"Completed runs: {len(generated_proteins)}/{self.num_runs}")
                print(f"Average cost: {avg_cost:.2f} ± {std_cost:.2f}")
                print(f"Average time: {avg_time:.2f}s ± {std_time:.2f}s")
                print(f"Global entropy: {global_entropy:.4f}" if global_entropy else "Global entropy: N/A")
                print(f"Average position-specific entropy: {avg_position_entropy:.4f}" if avg_position_entropy else "Average position entropy: N/A")
                print(f"Levenshtein diversity: {levenshtein_diversity:.4f}" if levenshtein_diversity else "Levenshtein diversity: N/A")
                print(f"Average metrics across proteins:")
                for metric, value in avg_single_metrics.items():
                    print(f"  - {metric}: {value:.4f}")
                
                # Write summary to log file as well
                with open(log_file, 'a') as main_log:
                    main_log.write("\n\n--- Summary ---\n")
                    main_log.write(f"Strategy: {strategy_name}\n")
                    main_log.write(f"Strategy parameters: {strategy_params}\n")
                    main_log.write(f"Completed runs: {len(generated_proteins)}/{self.num_runs}\n")
                    main_log.write(f"Average cost: {avg_cost:.2f} ± {std_cost:.2f}\n")
                    main_log.write(f"Average time: {avg_time:.2f}s ± {std_time:.2f}s\n")
                    if global_entropy:
                        main_log.write(f"Global entropy: {global_entropy:.4f}\n")
                    if avg_position_entropy:
                        main_log.write(f"Average position-specific entropy: {avg_position_entropy:.4f}\n")
                    if levenshtein_diversity:
                        main_log.write(f"Levenshtein diversity: {levenshtein_diversity:.4f}\n")
                    main_log.write(f"Average metrics across proteins:\n")
                    for metric, value in avg_single_metrics.items():
                        main_log.write(f"  - {metric}: {value:.4f}\n")
                
                # Store results
                self.results[strategy_name] = results
                
                # Save sequences and stats to job folder
                self._save_sequences(job_folder, generated_proteins, strategy_name)
                self._save_stats(job_folder, results, strategy_name)
                
                print(f"Parallel benchmark for {strategy_name} completed. Log saved to {log_file}")
                return results
                
            else:
                print("No proteins were successfully generated.")
                with open(log_file, 'a') as main_log:
                    main_log.write("\n\nNo proteins were successfully generated.\n")
                    
                results = {
                    "strategy": strategy_name,
                    "strategy_params": strategy_params,
                    "num_runs": self.num_runs,
                    "num_completed": 0,
                    "error": "No proteins were successfully generated."
                }
                
                # Save stats even if no proteins were generated
                self._save_stats(job_folder, results, strategy_name)
                
                return results
                
        except Exception as e:
            print(f"Parallel benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            
            with open(log_file, 'a') as main_log:
                main_log.write(f"\n\nParallel benchmark failed: {e}\n")
                main_log.write(traceback.format_exc())
            
            results = {
                "strategy": strategy_name,
                "strategy_params": strategy_params,
                "error": str(e),
                "num_completed": len(generated_proteins) if 'generated_proteins' in locals() else 0
            }
            
            # Save stats even on error
            self._save_stats(job_folder, results, strategy_name)
            
            return results
    
    def _average_single_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Average the single metrics across all proteins.
        
        Args:
            metrics_list: List of metric dictionaries for each protein
            
        Returns:
            dict: Dictionary of averaged metrics
        """
        if not metrics_list:
            return {}
            
        # Collect all metrics by name
        all_metrics = {}
        for metrics in metrics_list:
            for name, value in metrics.items():
                if name not in all_metrics:
                    all_metrics[name] = []
                all_metrics[name].append(value)
        
        # Average each metric
        avg_metrics = {}
        for name, values in all_metrics.items():
            try:
                avg_metrics[name] = np.mean(values)
            except Exception as e:
                print(f"Error averaging metric {name}: {e}")
                avg_metrics[name] = float('nan')
                
        return avg_metrics
            
def run_denoising_benchmarks(client, source_protein_path, strategy_instances, **kwargs):
    """
    Run benchmarks for multiple denoising strategies.
    
    Args:
        client: ESM3InferenceClient instance
        source_protein_path: Path to the source protein PDB file
        strategy_instances: List of (strategy_instance, strategy_name) tuples
        **kwargs: Additional arguments for BenchmarkRunner
        
    Returns:
        tuple: (BenchmarkRunner instance, comparison results)
    """
    # Create benchmark runner with source protein path
    runner = BenchmarkRunner(client, source_protein_path=source_protein_path, **kwargs)
    
    # Run benchmarks for each strategy
    for strategy, strategy_name in strategy_instances:
        runner.run_benchmark(strategy, strategy_name)
        
    # Compare results
    comparison = runner.compare_strategies()
    
    return runner, comparison

def save_benchmark_results(runner: BenchmarkRunner, output_file=None):
    """
    Save benchmark results to a file.
    
    Note: This function is maintained for backward compatibility.
    BenchmarkRunner now automatically saves results to the job folder.
    
    Args:
        runner: BenchmarkRunner instance with results
        output_file: Output file path (default: generates timestamped filename)
        
    Returns:
        str: Path to the saved file
    """
    import json
    from datetime import datetime
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"denoising_benchmark_results_{timestamp}.json"
        
    # Convert results to serializable format
    serializable_results = {}
    
    for strategy_name, result in runner.results.items():
        # Make a copy to avoid modifying the original
        serialized_result = {}
        
        # Selectively copy fields that can be serialized
        for key, value in result.items():
            # Skip non-serializable objects
            if key in ['generated_proteins', 'single_metrics']:
                continue
                
            if key == 'avg_single_metrics':
                # Convert any numpy values to regular Python types
                serialized_result[key] = {
                    metric: val.item() if hasattr(val, 'item') else val
                    for metric, val in value.items()
                }
                continue
                
            # Handle numpy values
            if hasattr(value, 'item'):
                serialized_result[key] = value.item()
            elif isinstance(value, dict):
                # Handle nested dictionaries
                serialized_result[key] = {}
                for k, v in value.items():
                    if hasattr(v, 'item'):
                        serialized_result[key][k] = v.item()
                    else:
                        serialized_result[key][k] = v
            elif isinstance(value, (list, np.ndarray)) and key in ["costs", "times"]:
                # Convert costs and times to lists
                serialized_result[key] = [x.item() if hasattr(x, 'item') else x for x in value]
            else:
                # Other simple types
                serialized_result[key] = value
                
        # Record counts instead of full sequences
        if 'generated_proteins' in result:
            serialized_result['num_generated'] = len(result['generated_proteins'])
                
        serializable_results[strategy_name] = serialized_result
        
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "num_runs": runner.num_runs
                },
                "results": serializable_results
            },
            f,
            indent=2
        )
        
    print(f"Benchmark results saved to {output_file}")
    return output_file 