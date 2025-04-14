from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from tqdm import tqdm
import attr
import random
import os
import sys
from datetime import datetime

from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    ESMProteinTensor,
    SamplingConfig,
    SamplingTrackConfig,
    LogitsConfig,
)
from esm.models.esm3 import ESM3
from esm.tokenization import get_esm3_model_tokenizers

# Ensure TOKENIZERS_PATH environment variable is set or adjust path as needed
# Assuming default installation location for tokenizers relative to esm package
default_tokenizers_path = os.path.join(os.path.dirname(__file__), "..", "tokenizers")
os.environ["TOKENIZERS_PATH"] = os.environ.get("TOKENIZERS_PATH", default_tokenizers_path)

# Add a function to print to both console and file
class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout
        
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        
    def flush(self):
        self.stdout.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()

class PrintFormatter:
    """Utility class to handle hierarchical printing with proper indentation."""
    
    def __init__(self):
        self.indent_level = 0
        self.indent_char = "│   "
        self.branch_char = "├── "
        self.last_char = "└── "
        
    def print(self, message: str, is_last: bool = False, increase_indent: bool = False, decrease_indent: bool = False):
        """Print a message with proper indentation and tree structure."""
        if decrease_indent:
            self.indent_level = max(0, self.indent_level - 1)
            
        prefix = self.indent_char * self.indent_level
        if self.indent_level > 0:
            prefix = prefix[:-4] + (self.last_char if is_last else self.branch_char)
            
        print(f"{prefix}{message}")
        
        if increase_indent:
            self.indent_level += 1

class BaseDenoising(ABC):
    """Abstract base class for denoising strategies."""
    
    def __init__(self, client: ESM3InferenceClient):
        if isinstance(client, ESM3):
            self.tokenizers = client.tokenizers
        else:
            raise ValueError("client must be an instance of ESM3")
        self.client = client
        self.track = "sequence"  # Default track
        self.printer = PrintFormatter()  # Add printer
        self.cost = 0  # Number of calls to the model
        
    def add_noise(
        self,
        protein_tensor: ESMProteinTensor,
        noise_percentage: float,
        track: str = "sequence"
    ) -> ESMProteinTensor:
        """Add noise by masking random positions."""
        track_tensor = getattr(protein_tensor, track)
        track_tokenizer = getattr(self.tokenizers, track)
        
        self.printer.print("Adding noise to protein tensor", increase_indent=True)
        
        # Calculate number of positions to mask
        num_positions = len(track_tensor)
        valid_positions = list(range(1, num_positions - 1))
        num_to_mask = int(len(valid_positions) * (noise_percentage / 100))
        
        # Randomly select positions to mask
        random.shuffle(valid_positions)
        mask_positions = valid_positions[:num_to_mask]
        
        # Create new tensor with masked positions
        protein_tensor = attr.evolve(protein_tensor)
        track_tensor = track_tensor.clone()
        track_tensor[mask_positions] = track_tokenizer.mask_token_id
        setattr(protein_tensor, track, track_tensor)
        
        self.printer.print(f"Masked positions ({track}): {mask_positions}")
        self.printer.print(f"Resulting tensor: {getattr(protein_tensor, track)}", 
                          is_last=True, decrease_indent=True)
        return protein_tensor
    
    def maybe_add_default_structure_tokens(
        self, protein_tensor: ESMProteinTensor
    ) -> ESMProteinTensor:
        """Add default structure tokens if needed."""
        empty_protein_tensor = ESMProteinTensor.empty(
            len(protein_tensor) - 2,
            tokenizers=self.tokenizers,
            device=protein_tensor.device,
        )
        if protein_tensor.structure is None:
            setattr(protein_tensor, "structure", empty_protein_tensor.structure)
        else:
            print("Warning: structure already exists in protein_tensor")
        return protein_tensor

    def get_number_of_masked_positions(
        self, protein_tensor: ESMProteinTensor, track: str = "sequence"
    ) -> int:
        """Get number of masked positions, excluding start and end tokens."""
        track_tensor = getattr(protein_tensor, track)
        track_tokenizer = getattr(self.tokenizers, track)
        is_mask = track_tensor == track_tokenizer.mask_token_id
        return is_mask[1:-1].sum().item()
    
    def unmask_position(
        self,
        protein_tensor: ESMProteinTensor,
        position: int,
        temperature: float,
        track: str = "sequence",
        model_output = None
    ) -> ESMProteinTensor:
        """Unmask a specific position using the model's prediction."""
        self.printer.print(f"Unmasking position {position}", increase_indent=True)
        
        track_tensor = getattr(protein_tensor, track)
        track_tokenizer = getattr(self.tokenizers, track)
        
        protein_tensor = attr.evolve(protein_tensor)
        track_tensor = track_tensor.clone()
        
        if model_output is None:
            sampling_config = SamplingConfig()
            setattr(sampling_config, track, SamplingTrackConfig(temperature=temperature))
            
            output = self.client.forward_and_sample(
                protein_tensor,
                sampling_configuration=sampling_config
            )
            self.cost += 1
            assert not isinstance(output, ESMProteinError)
        else:
            output = model_output
        
        output_track_tensor = getattr(output.protein_tensor, track)
        predicted_token_id = output_track_tensor[position].item()
        predicted_token = getattr(self.tokenizers, self.track).decode(torch.tensor([predicted_token_id]))
        
        track_tensor[position] = output_track_tensor[position]
        setattr(protein_tensor, track, track_tensor)
        
        self.printer.print(f"Predicted token: {predicted_token} ({predicted_token_id})", 
                          is_last=True, decrease_indent=True)
        return protein_tensor
    
    @abstractmethod
    def get_next_position(
        self,
        protein_tensor: ESMProteinTensor,
        track: str = "sequence",
        model_output = None
    ) -> int:
        """Get the next position to unmask based on the strategy."""
        pass
    
    @abstractmethod
    def denoise(
        self,
        protein: ESMProtein,
        noise_percentage: float,
        num_decoding_steps: int,
        temperature: float = 0.0,
        track: str = "sequence",
        verbose: bool = True,
    ) -> ESMProtein:
        """Denoise a protein sequence using the specific strategy."""
        pass

class EntropyBasedDenoising(BaseDenoising):
    """Denoising strategy that selects positions with lowest entropy."""
    
    def get_next_position(
        self,
        protein_tensor: ESMProteinTensor,
        track: str = "sequence",
        model_output = None
    ) -> int:
        """Get next position to unmask based on entropy."""
        self.printer.print("Computing position entropies", increase_indent=True)
        
        if model_output is None:
            output = self.client.forward_and_sample(
                protein_tensor,
                sampling_configuration=SamplingConfig(
                    sequence=SamplingTrackConfig(temperature=1.0, topk_logprobs=5),
                    structure=SamplingTrackConfig(temperature=1.0, topk_logprobs=5),
                )
            )
            self.cost += 1
            assert not isinstance(output, ESMProteinError)
        else:
            output = model_output
        
        # Get entropy directly from output
        entropy = getattr(output.entropy, track)
        self.printer.print(f"Raw entropies: {entropy}")
        
        # Mask entropy of already unmasked positions
        track_tensor = getattr(protein_tensor, track)
        track_tokenizer = getattr(self.tokenizers, track)
        is_mask = track_tensor == track_tokenizer.mask_token_id
        
        # Set entropy to inf for non-masked positions and start/end tokens
        entropy[~is_mask] = float('inf')
        entropy[0] = float('inf')
        entropy[-1] = float('inf')
        
        self.printer.print(f"Masked positions entropies: {entropy}", is_last=True, decrease_indent=True)
        return entropy.argmin().item()
        
    def denoise(
        self,
        protein: ESMProtein,
        noise_percentage: float,
        num_decoding_steps: int,
        temperature: float = 0.0,
        track: str = "sequence",
        verbose: bool = True,
    ) -> ESMProtein:
        """Denoise using entropy-based strategy."""
        self.printer.print("Starting entropy-based denoising process", increase_indent=True)
        
        # Encode protein and add noise
        protein_tensor = self.client.encode(protein)
        assert not isinstance(protein_tensor, ESMProteinError)
        
        if track == "structure":
            protein_tensor = self.maybe_add_default_structure_tokens(protein_tensor)
            
        # Add noise by masking positions
        protein_tensor = self.add_noise(protein_tensor, noise_percentage, track)
        
        # Print initial sequence
        initial_decoded = self.client.decode(protein_tensor)
        assert not isinstance(initial_decoded, ESMProteinError)
        self.printer.print(f"Initial sequence: {initial_decoded.sequence}")
        
        self.printer.print("Starting denoising steps:", increase_indent=True)
        if verbose:
            pbar = tqdm(range(num_decoding_steps), desc="Denoising")
        else:
            pbar = range(num_decoding_steps)
            
        for step in pbar:
            is_last_step = step == num_decoding_steps - 1
            self.printer.print(f"Step {step+1}/{num_decoding_steps}", increase_indent=True)
            
            # Get model output once per step
            output = self.client.forward_and_sample(
                protein_tensor,
                sampling_configuration=SamplingConfig(
                    sequence=SamplingTrackConfig(temperature=1.0, topk_logprobs=5),
                    structure=SamplingTrackConfig(temperature=1.0, topk_logprobs=5),
                )
            )
            self.cost += 1
            assert not isinstance(output, ESMProteinError)
            
            # Get next position using the same model output
            next_pos = self.get_next_position(protein_tensor, track, output)
            
            # Unmask using the same model output
            protein_tensor = self.unmask_position(
                protein_tensor, next_pos, temperature, track, output
            )
            
            if verbose:
                after_unmask_decoded = self.client.decode(protein_tensor)
                assert not isinstance(after_unmask_decoded, ESMProteinError)
                self.printer.print(f"Current sequence: {after_unmask_decoded.sequence}", 
                                 is_last=True, decrease_indent=True)
        
        # Final prediction
        protein_tensor_output = self.client.forward_and_sample(
            protein_tensor,
            SamplingConfig(
                sequence=SamplingTrackConfig(temperature=temperature),
                structure=SamplingTrackConfig(temperature=temperature),
            ),
        )
        self.cost += 1
        
        assert not isinstance(protein_tensor_output, ESMProteinError)
        protein_tensor = protein_tensor_output.protein_tensor
        
        decoded_protein = self.client.decode(protein_tensor)
        assert not isinstance(decoded_protein, ESMProteinError)
        self.printer.print(f"Final denoised sequence: {decoded_protein.sequence}")
        self.printer.print(f"Total model calls: {self.cost}", is_last=True, decrease_indent=True)
        return decoded_protein

class MaxProbBasedDenoising(BaseDenoising):
    """Denoising strategy that selects positions with highest probability for any token."""
    
    def get_next_position(
        self,
        protein_tensor: ESMProteinTensor,
        track: str = "sequence",
        model_output = None
    ) -> int:
        """Get next position to unmask based on maximum probability."""
        self.printer.print("Computing position probabilities", increase_indent=True)
        
        if model_output is None:
            output = self.client.forward_and_sample(
                protein_tensor,
                sampling_configuration=SamplingConfig(
                    sequence=SamplingTrackConfig(temperature=1.0, topk_logprobs=5),
                    structure=SamplingTrackConfig(temperature=1.0, topk_logprobs=5),
                )
            )
            self.cost += 1
            assert not isinstance(output, ESMProteinError)
        else:
            output = model_output
        
        max_probs = getattr(output.top_prob, track)
        self.printer.print(f"Raw probabilities: {max_probs}")
        
        # Mask probabilities of already unmasked positions
        track_tensor = getattr(protein_tensor, track)
        track_tokenizer = getattr(self.tokenizers, track)
        is_mask = track_tensor == track_tokenizer.mask_token_id
        
        # Set probability to -inf for non-masked positions and start/end tokens
        max_probs[~is_mask] = -float('inf')  # Non-masked positions
        max_probs[0] = -float('inf')  # Start token
        max_probs[-1] = -float('inf')  # End token
        
        self.printer.print(f"Masked positions probabilities: {max_probs}", is_last=True, decrease_indent=True)
        return max_probs.argmax().item()
        
    def denoise(
        self,
        protein: ESMProtein,
        noise_percentage: float,
        num_decoding_steps: int,
        temperature: float = 0.0,
        track: str = "sequence",
        verbose: bool = True,
    ) -> ESMProtein:
        """Denoise using max-probability strategy."""
        self.printer.print("Starting max-probability denoising process", increase_indent=True)
        
        # Encode protein and add noise
        protein_tensor = self.client.encode(protein)
        assert not isinstance(protein_tensor, ESMProteinError)
        
        if track == "structure":
            protein_tensor = self.maybe_add_default_structure_tokens(protein_tensor)
            
        # Add noise by masking positions
        protein_tensor = self.add_noise(protein_tensor, noise_percentage, track)
        
        # Print initial sequence
        initial_decoded = self.client.decode(protein_tensor)
        assert not isinstance(initial_decoded, ESMProteinError)
        self.printer.print(f"Initial sequence: {initial_decoded.sequence}")
        
        self.printer.print("Starting denoising steps:", increase_indent=True)
        if verbose:
            pbar = tqdm(range(num_decoding_steps), desc="Denoising")
        else:
            pbar = range(num_decoding_steps)
            
        for step in pbar:
            is_last_step = step == num_decoding_steps - 1
            self.printer.print(f"Step {step+1}/{num_decoding_steps}", increase_indent=True)
            
            # Get model output once per step
            output = self.client.forward_and_sample(
                protein_tensor,
                sampling_configuration=SamplingConfig(
                    sequence=SamplingTrackConfig(temperature=1.0, topk_logprobs=5),
                    structure=SamplingTrackConfig(temperature=1.0, topk_logprobs=5),
                )
            )
            self.cost += 1
            assert not isinstance(output, ESMProteinError)
            
            # Get next position using the same model output
            next_pos = self.get_next_position(protein_tensor, track, output)
            
            # Unmask using the same model output
            protein_tensor = self.unmask_position(
                protein_tensor, next_pos, temperature, track, output
            )
            
            if verbose:
                after_unmask_decoded = self.client.decode(protein_tensor)
                assert not isinstance(after_unmask_decoded, ESMProteinError)
                self.printer.print(f"Current sequence: {after_unmask_decoded.sequence}", 
                                 is_last=True, decrease_indent=True)
        
        # Final prediction
        protein_tensor_output = self.client.forward_and_sample(
            protein_tensor,
            SamplingConfig(
                sequence=SamplingTrackConfig(temperature=temperature),
                structure=SamplingTrackConfig(temperature=temperature),
            ),
        )
        self.cost += 1
        
        assert not isinstance(protein_tensor_output, ESMProteinError)
        protein_tensor = protein_tensor_output.protein_tensor
        
        decoded_protein = self.client.decode(protein_tensor)
        assert not isinstance(decoded_protein, ESMProteinError)
        self.printer.print(f"Final denoised sequence: {decoded_protein.sequence}")
        self.printer.print(f"Total model calls: {self.cost}", is_last=True, decrease_indent=True)
        return decoded_protein

# Added test case
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = "denoising_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamped output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"denoising_output_{timestamp}.txt")
    
    # Redirect output to both console and file
    tee = Tee(output_file)
    sys.stdout = tee
    
    try:
        # --- Configuration ---
        MODEL_NAME = "esm3_sm_8M_v1" # Or another available model
        TEST_SEQUENCE = "ACDE"
        NOISE_PERCENTAGE = 50.0 # Mask 50% initially (2 positions for length 4)
        NUM_DECODING_STEPS = 2 # Number of steps to unmask
        TEMPERATURE = 0.0
        TRACK = "sequence"
        # --- End Configuration ---

        # Check for GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}\n")

        # Load local ESM3 model
        model = ESM3.from_pretrained()
        model.to(device)
        model.eval()

        # Create a dummy protein
        protein = ESMProtein(sequence=TEST_SEQUENCE)
        print(f"Original Protein: {protein.sequence}\n")

        # Instantiate Denoiser with local model
        denoiser = EntropyBasedDenoising(model)
        denoiser.track = TRACK # Set track for prints

        # Run denoising
        denoised_protein = denoiser.denoise(
            protein=protein,
            noise_percentage=NOISE_PERCENTAGE,
            num_decoding_steps=NUM_DECODING_STEPS,
            temperature=TEMPERATURE,
            track=TRACK,
            verbose=True
        )

        # Try MaxProbBasedDenoising
        print("\n=== Testing MaxProbBasedDenoising ===")
        max_prob_denoiser = MaxProbBasedDenoising(model)
        max_prob_denoiser.track = TRACK

        denoised_protein_maxprob = max_prob_denoiser.denoise(
            protein=protein,
            noise_percentage=NOISE_PERCENTAGE,
            num_decoding_steps=NUM_DECODING_STEPS,
            temperature=TEMPERATURE,
            track=TRACK,
            verbose=True
        )
        
    except ImportError:
        print("Error: Could not import ESM3 model.")
        print("Please ensure you have the correct version of esm installed.")
    except Exception as e:
        print(f"An error occurred during the test run: {e}")
    finally:
        # Restore stdout and close the file
        sys.stdout = tee.stdout
        tee.close()
        print(f"\nOutput has been saved to: {output_file}")
