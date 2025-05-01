from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    LogitsConfig
)
import torch.nn.functional as F
import math  # Add import for math

### Metrics
def UACCE(client: ESM3InferenceClient, protein: ESMProtein, verbose: bool = False) -> float:
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


### Benchmarking run
def get_smallest_pdb_file():
    pass