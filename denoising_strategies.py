from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from tqdm import tqdm
import attr
import random
import os
import sys
from datetime import datetime
from typing import Tuple

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
from esm.sdk.forge import ESM3ForgeInferenceClient

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

class BaseDenoisingStrategy(ABC):
    """Abstract base class for denoising strategies."""
    
    def __init__(
        self, 
        client: ESM3InferenceClient,
        noise_percentage: float = 50.0,
        num_decoding_steps: int = 20,  # Changed default to 20
        temperature: float = 0.0,
        track: str = "sequence"
    ):
        if isinstance(client, ESM3):
            self.tokenizers = client.tokenizers
        elif isinstance(client, ESM3ForgeInferenceClient):
            self.tokenizers = get_esm3_model_tokenizers(client.model)
        else:
            raise ValueError(
                "client must be an instance of ESM3 or ESM3ForgeInferenceClient"
            )

        self.client = client
        self.track = track  # Default track
        self.printer = PrintFormatter()  # Add printer
        self.cost = 0  # Number of calls to the model
        self.protein = None  # Placeholder for the protein object
        
        # Store denoising parameters
        self.noise_percentage = noise_percentage
        self.num_decoding_steps = num_decoding_steps
        self.temperature = temperature
        
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
    
    def unmask_positions(
        self,
        protein_tensor: ESMProteinTensor,
        positions: list,
        temperature: float,
        track: str = "sequence",
        model_output = None
    ) -> ESMProteinTensor:
        """Unmask multiple specific positions using the model's prediction."""
        self.printer.print(f"Unmasking positions {positions}", increase_indent=True)
        
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
        
        for position in positions:
            predicted_token_id = int(output_track_tensor[position].item())
            predicted_token = getattr(self.tokenizers, self.track).decode(torch.tensor([predicted_token_id]))
            
            track_tensor[position] = output_track_tensor[position]
            self.printer.print(f"Position {position} - Predicted token: {predicted_token} ({predicted_token_id})")
        
        setattr(protein_tensor, track, track_tensor)
        self.printer.print("Unmasking complete", is_last=True, decrease_indent=True)
        return protein_tensor

    def calculate_positions_per_step(
        self, 
        total_masked_positions: int,
        max_steps: int
    ) -> int:
        """Calculate how many positions to unmask per step."""
        import math
        return math.ceil(total_masked_positions / max_steps)
    
    def unmask_position(
        self,
        protein_tensor: ESMProteinTensor,
        position: int,
        temperature: float,
        track: str = "sequence",
        model_output = None
    ) -> ESMProteinTensor:
        """Unmask a specific position using the model's prediction."""
        return self.unmask_positions(protein_tensor, [position], temperature, track, model_output)
    
    def return_generation(
        self,
    ) -> Tuple[ESMProtein, int]:
        """
        Return function for generation.
        Returns the following:
        - protein: ESMProtein
        - cost: int
        """
        return [self.protein, self.cost]

    @abstractmethod
    def get_next_positions(
        self,
        protein_tensor: ESMProteinTensor,
        num_positions: int,
        track: str = "sequence",
        model_output = None
    ) -> list:
        """Get the next positions to unmask based on the strategy."""
        pass
    
    def get_next_position(
        self,
        protein_tensor: ESMProteinTensor,
        track: str = "sequence",
        model_output = None
    ) -> int:
        """Backward compatibility method for getting single position."""
        return self.get_next_positions(protein_tensor, 1, track, model_output)[0]
    
    @abstractmethod
    def denoise(
        self,
        protein: ESMProtein,
        verbose: bool = True,
    ) -> ESMProtein:
        """Denoise a protein sequence using the specific strategy."""
        pass

    def _extract_strategy_params(self) -> dict:
        """Extract base parameters common to all denoising strategies."""
        return {
            "noise_percentage": self.noise_percentage,
            "num_decoding_steps": self.num_decoding_steps,
            "temperature": self.temperature,
            "track": self.track
        }

class EntropyBasedDenoising(BaseDenoisingStrategy):
    """Denoising strategy that selects positions with lowest entropy."""
    
    def get_next_positions(
        self,
        protein_tensor: ESMProteinTensor,
        num_positions: int,
        track: str = "sequence",
        model_output = None
    ) -> list:
        """Get multiple positions to unmask based on entropy."""
        self.printer.print(f"Computing position entropies for {num_positions} positions", increase_indent=True)
        
        if model_output is None:
            output = self.client.forward_and_sample(
                protein_tensor,
                sampling_configuration=SamplingConfig(
                    sequence=SamplingTrackConfig(temperature=self.temperature, topk_logprobs=5),
                    structure=SamplingTrackConfig(temperature=self.temperature, topk_logprobs=5),
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
        
        self.printer.print(f"Masked positions entropies: {entropy}")
        
        # Get positions with lowest entropy
        positions = []
        entropy_copy = entropy.clone()
        
        for i in range(min(num_positions, is_mask.sum().item())):
            next_pos = entropy_copy.argmin().item()
            positions.append(next_pos)
            entropy_copy[next_pos] = float('inf')  # Mark as processed
            
        self.printer.print(f"Selected positions: {positions}", is_last=True, decrease_indent=True)
        return positions
    
    def get_next_position(
        self,
        protein_tensor: ESMProteinTensor,
        track: str = "sequence",
        model_output = None
    ) -> int:
        """Get next position to unmask based on entropy (backward compatibility)."""
        return self.get_next_positions(protein_tensor, 1, track, model_output)[0]
        
    def denoise(
        self,
        protein: ESMProtein,
        verbose: bool = True,
        max_decoding_steps: int = 20,
    ) -> ESMProtein:
        """Denoise using entropy-based strategy."""
        self.printer.print("Starting entropy-based denoising process", increase_indent=True)
        
        # Encode protein and add noise
        protein_tensor = self.client.encode(protein)
        assert not isinstance(protein_tensor, ESMProteinError)
        
        if self.track == "structure":
            protein_tensor = self.maybe_add_default_structure_tokens(protein_tensor)
            
        # Add noise by masking positions
        protein_tensor = self.add_noise(protein_tensor, self.noise_percentage, self.track)
        
        # Calculate masked positions
        total_masked = self.get_number_of_masked_positions(protein_tensor, self.track)
        self.printer.print(f"Total masked positions: {total_masked}")
        
        # Calculate how many positions to unmask per step
        max_steps = min(max_decoding_steps, self.num_decoding_steps)
        positions_per_step = self.calculate_positions_per_step(total_masked, max_steps)
        actual_steps = min((total_masked + positions_per_step - 1) // positions_per_step, max_steps)
        
        self.printer.print(f"Positions per step: {positions_per_step}")
        self.printer.print(f"Actual number of steps: {actual_steps}")
        
        # Print initial sequence
        initial_decoded = self.client.decode(protein_tensor)
        assert not isinstance(initial_decoded, ESMProteinError)
        self.printer.print(f"Initial sequence: {initial_decoded.sequence}")
        
        self.printer.print("Starting denoising steps:", increase_indent=True)
        if verbose:
            pbar = tqdm(range(actual_steps), desc="Denoising")
        else:
            pbar = range(actual_steps)
            
        remaining_masked = total_masked
        
        for step in pbar:
            is_last_step = step == actual_steps - 1
            self.printer.print(f"Step {step+1}/{actual_steps}", increase_indent=True)
            
            # On last step, handle any remaining positions
            positions_this_step = min(positions_per_step, remaining_masked)
            if is_last_step:
                positions_this_step = remaining_masked
                
            # Get model output once per step
            output = self.client.forward_and_sample(
                protein_tensor,
                sampling_configuration=SamplingConfig(
                    sequence=SamplingTrackConfig(temperature=self.temperature, topk_logprobs=5),
                    structure=SamplingTrackConfig(temperature=self.temperature, topk_logprobs=5),
                )
            )
            self.cost += 1
            assert not isinstance(output, ESMProteinError)
            
            # Get next positions using the same model output
            next_positions = self.get_next_positions(
                protein_tensor, positions_this_step, self.track, output
            )
            
            # Unmask using the same model output
            protein_tensor = self.unmask_positions(
                protein_tensor, next_positions, self.temperature, self.track, output
            )
            
            # Update remaining masked count
            remaining_masked -= len(next_positions)
            
            if verbose:
                after_unmask_decoded = self.client.decode(protein_tensor)
                assert not isinstance(after_unmask_decoded, ESMProteinError)
                self.printer.print(f"Current sequence: {after_unmask_decoded.sequence}", 
                                 is_last=True, decrease_indent=True)
        
        # Final prediction
        protein_tensor_output = self.client.forward_and_sample(
            protein_tensor,
            SamplingConfig(
                sequence=SamplingTrackConfig(temperature=self.temperature),
                structure=SamplingTrackConfig(temperature=self.temperature),
            ),
        )
        self.cost += 1
        
        assert not isinstance(protein_tensor_output, ESMProteinError)
        protein_tensor = protein_tensor_output.protein_tensor
        
        decoded_protein = self.client.decode(protein_tensor)
        assert not isinstance(decoded_protein, ESMProteinError)
        self.printer.print(f"Final denoised sequence: {decoded_protein.sequence}")
        self.printer.print(f"Total model calls: {self.cost}", is_last=True, decrease_indent=True)
        self.protein = decoded_protein
        return decoded_protein
        
    def _extract_strategy_params(self) -> dict:
        """Extract strategy parameters specific to EntropyBasedDenoising."""
        params = super()._extract_strategy_params()
        # Add any EntropyBasedDenoising-specific parameters here
        return params

class MaxProbBasedDenoising(BaseDenoisingStrategy):
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
                    sequence=SamplingTrackConfig(temperature=1.0, top_p=1.0),
                    structure=SamplingTrackConfig(temperature=1.0, top_p=1.0),
                )
            )
            self.cost += 1
            assert not isinstance(output, ESMProteinError)
        else:
            output = model_output
        print("Output:", output)
        max_probs = getattr(output.top_prob, track)
        self.printer.print(f"Top probabilities at each position: {max_probs}")
        
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
        verbose: bool = True,
    ) -> ESMProtein:
        """Denoise using max-probability strategy."""
        self.printer.print("Starting max-probability denoising process", increase_indent=True)
        
        # Encode protein and add noise
        protein_tensor = self.client.encode(protein)
        assert not isinstance(protein_tensor, ESMProteinError)
        
        if self.track == "structure":
            protein_tensor = self.maybe_add_default_structure_tokens(protein_tensor)
            
        # Add noise by masking positions
        protein_tensor = self.add_noise(protein_tensor, self.noise_percentage, self.track)
        
        # Print initial sequence
        initial_decoded = self.client.decode(protein_tensor)
        assert not isinstance(initial_decoded, ESMProteinError)
        self.printer.print(f"Initial sequence: {initial_decoded.sequence}")
        
        self.printer.print("Starting denoising steps:", increase_indent=True)
        if verbose:
            pbar = tqdm(range(self.num_decoding_steps), desc="Denoising")
        else:
            pbar = range(self.num_decoding_steps)
            
        for step in pbar:
            is_last_step = step == self.num_decoding_steps - 1
            self.printer.print(f"Step {step+1}/{self.num_decoding_steps}", increase_indent=True)
            
            # Get model output once per step
            output = self.client.forward_and_sample(
                protein_tensor,
                sampling_configuration=SamplingConfig(
                    sequence=SamplingTrackConfig(temperature=1.0, top_p=1.0),
                    structure=SamplingTrackConfig(temperature=1.0, top_p=1.0),
                )
            )
            self.cost += 1
            assert not isinstance(output, ESMProteinError)
            
            # Get next position using the same model output
            next_pos = self.get_next_position(protein_tensor, self.track, output)
            
            # Unmask using the same model output
            protein_tensor = self.unmask_position(
                protein_tensor, next_pos, self.temperature, self.track, output
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
                sequence=SamplingTrackConfig(temperature=self.temperature),
                structure=SamplingTrackConfig(temperature=self.temperature),
            ),
        )
        self.cost += 1
        
        assert not isinstance(protein_tensor_output, ESMProteinError)
        protein_tensor = protein_tensor_output.protein_tensor
        
        decoded_protein = self.client.decode(protein_tensor)
        assert not isinstance(decoded_protein, ESMProteinError)
        self.printer.print(f"Final denoised sequence: {decoded_protein.sequence}")
        self.printer.print(f"Total model calls: {self.cost}", is_last=True, decrease_indent=True)
        self.protein = decoded_protein
        return decoded_protein
        
    def _extract_strategy_params(self) -> dict:
        """Extract strategy parameters specific to MaxProbBasedDenoising."""
        params = super()._extract_strategy_params()
        # Add any MaxProbBasedDenoising-specific parameters here
        return params

