from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from tqdm import tqdm
import attr
import random
import os
import sys
from datetime import datetime
from typing import Tuple, Callable, Optional, Dict, Any, List, Union

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
from esm.sdk.forge import ESM3ForgeInferenceClient

# Hardcoded token IDs
MASK_TOKEN_ID = 32  # Mask token ID for sequence

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
    
    @classmethod
    def get_required_params(cls):
        """
        Return the list of parameters required by this strategy.
        
        Each subclass should override this to specify its required parameters.
        This allows automatic parameter filtering and strategy creation.
        
        Returns:
            set: Set of parameter names required by this strategy
        """
        # Base parameters required by all strategies
        return {'client', 'noise_percentage', 'num_decoding_steps'}
    
    @classmethod
    def create(cls, **params):
        """
        Factory method to create a strategy instance with filtered parameters.
        
        This method automatically filters parameters to only include those
        required by the specific strategy class, allowing safe instantiation
        without parameter conflicts.
        
        Args:
            **params: All available parameters
            
        Returns:
            BaseDenoisingStrategy: An instance of the specific strategy
        """
        required_params = cls.get_required_params()
        filtered_params = {k: v for k, v in params.items() if k in required_params}
        return cls(**filtered_params)
    
    def __init__(
        self, 
        client: ESM3InferenceClient,
        noise_percentage: float = 50.0,
        num_decoding_steps: int = 20,
        temperature: float = 0.0,
    ):
        self.client = client
        self.printer = PrintFormatter()
        self.cost = 0  # Number of calls to the model
        self.protein = None  # Placeholder for the protein object
        
        # Store denoising parameters
        self.noise_percentage = noise_percentage
        self.num_decoding_steps = num_decoding_steps
        self.temperature = temperature
        
    def add_noise(
        self,
        protein_tensor: ESMProteinTensor,
        noise_percentage: float
    ) -> ESMProteinTensor:
        """Add noise by masking random positions in the sequence."""
        self.printer.print("Adding noise to protein tensor", increase_indent=True)
        
        # Calculate number of positions to mask
        sequence_tensor = protein_tensor.sequence
        num_positions = len(sequence_tensor)
        valid_positions = list(range(1, num_positions - 1))  # Skip start and end tokens
        num_to_mask = int(len(valid_positions) * (noise_percentage / 100))
        
        # Randomly select positions to mask
        random.shuffle(valid_positions)
        mask_positions = valid_positions[:num_to_mask]
        
        # Create new tensor with masked positions
        protein_tensor = attr.evolve(protein_tensor)
        sequence_tensor = sequence_tensor.clone()
        sequence_tensor[mask_positions] = MASK_TOKEN_ID
        protein_tensor.sequence = sequence_tensor
        
        self.printer.print(f"Masked positions: {mask_positions}")
        self.printer.print(f"Resulting tensor: {protein_tensor.sequence}", 
                          is_last=True, decrease_indent=True)
        return protein_tensor
    
    def get_number_of_masked_positions(
        self, protein_tensor: ESMProteinTensor
    ) -> int:
        """Get number of masked positions, excluding start and end tokens."""
        sequence_tensor = protein_tensor.sequence
        is_mask = sequence_tensor == MASK_TOKEN_ID
        return is_mask[1:-1].sum().item()
    
    def unmask_positions(
        self,
        protein_tensor: ESMProteinTensor,
        positions: list,
        temperature: float,
        model_output = None
    ) -> ESMProteinTensor:
        """Unmask multiple specific positions using the model's prediction."""
        self.printer.print(f"Unmasking positions {positions}", increase_indent=True)
        
        sequence_tensor = protein_tensor.sequence
        protein_tensor = attr.evolve(protein_tensor)
        sequence_tensor = sequence_tensor.clone()
        
        if model_output is None:
            sampling_config = SamplingConfig(
                sequence=SamplingTrackConfig(temperature=temperature)
            )
            
            output = self.client.forward_and_sample(
                protein_tensor,
                sampling_configuration=sampling_config
            )
            self.cost += 1
            assert not isinstance(output, ESMProteinError)
        else:
            output = model_output
        
        output_sequence_tensor = output.protein_tensor.sequence
        
        for position in positions:
            predicted_token_id = int(output_sequence_tensor[position].item())
            sequence_tensor[position] = output_sequence_tensor[position]
            self.printer.print(f"Position {position} - Predicted token ID: {predicted_token_id}")
        
        protein_tensor.sequence = sequence_tensor
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
        model_output = None
    ) -> ESMProteinTensor:
        """Unmask a specific position using the model's prediction."""
        return self.unmask_positions(protein_tensor, [position], temperature, model_output)
    
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

    def get_temperature(self, protein_tensor: ESMProteinTensor) -> float:
        """
        Get the temperature to use for the current step.
        Default implementation returns the fixed temperature.
        Subclasses can override for dynamic temperature strategies.
        """
        return self.temperature
    
    @abstractmethod
    def get_next_positions(
        self,
        protein_tensor: ESMProteinTensor,
        num_positions: int,
        model_output = None
    ) -> list:
        """Get the next positions to unmask based on the strategy."""
        pass
    
    def get_next_position(
        self,
        protein_tensor: ESMProteinTensor,
        model_output = None
    ) -> int:
        """Get the next position to unmask based on the strategy."""
        positions = self.get_next_positions(protein_tensor, 1, model_output)
        return positions[0]
    
    def denoise(
        self,
        protein: ESMProtein,
        verbose: bool = True,
        max_decoding_steps: int = None,
    ) -> ESMProtein:
        """
        Denoise a protein sequence using the strategy.
        
        Args:
            protein: The protein to denoise
            verbose: Whether to print progress
            max_decoding_steps: Maximum number of decoding steps (overrides num_decoding_steps)
            
        Returns:
            The denoised protein
        """
        self.protein = protein
        self.cost = 0
        
        if max_decoding_steps is not None:
            self.num_decoding_steps = max_decoding_steps
            
        # Add noise to the protein
        protein_tensor = self.add_noise(protein.protein_tensor, self.noise_percentage)
        
        # Get initial number of masked positions
        num_masked = self.get_number_of_masked_positions(protein_tensor)
        if num_masked == 0:
            return protein
            
        # Calculate positions to unmask per step
        positions_per_step = self.calculate_positions_per_step(
            num_masked, self.num_decoding_steps
        )
        
        # Main denoising loop
        for step in range(self.num_decoding_steps):
            if verbose:
                print(f"\nStep {step + 1}/{self.num_decoding_steps}")
                
            # Get next positions to unmask
            positions = self.get_next_positions(protein_tensor, positions_per_step)
            
            # Get temperature for this step
            temperature = self.get_temperature(protein_tensor)
            
            # Unmask the positions
            protein_tensor = self.unmask_positions(
                protein_tensor, positions, temperature
            )
            
            # Check if we're done
            num_masked = self.get_number_of_masked_positions(protein_tensor)
            if num_masked == 0:
                break
                
        # Create final protein
        final_protein = ESMProtein(protein_tensor)
        self.protein = final_protein
        
        return final_protein
    
    def _extract_strategy_params(self) -> dict:
        """Extract strategy-specific parameters for logging."""
        return {
            'noise_percentage': self.noise_percentage,
            'num_decoding_steps': self.num_decoding_steps,
            'temperature': self.temperature,
        } 