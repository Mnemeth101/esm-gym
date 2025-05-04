import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Dict, Any, Optional
from esm.sdk.api import ESM3InferenceClient, ESMProtein, ESMProteinError, LogitsConfig, GenerationConfig
from esm.sdk.forge import ESM3ForgeInferenceClient
import os

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