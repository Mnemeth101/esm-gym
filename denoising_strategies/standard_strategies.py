from .base import BaseDenoisingStrategy
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from esm.sdk.api import ESM3InferenceClient, ESMProteinTensor, SamplingConfig, SamplingTrackConfig, ESMProteinError, ESMProtein
import math
from tqdm import tqdm

class EntropyBasedDenoising(BaseDenoisingStrategy):
    """Denoising strategy that selects positions based on entropy of predictions."""
    
    def __init__(
        self,
        client: ESM3InferenceClient,
        noise_percentage: float = 50.0,
        num_decoding_steps: int = 20,
        temperature: float = 0.0,
    ):
        super().__init__(client, noise_percentage, num_decoding_steps, temperature)
    
    def get_next_positions(
        self,
        protein_tensor: ESMProteinTensor,
        num_positions: int,
        model_output = None,
        verbose: bool = True # Add verbose parameter
    ) -> List[int]:
        """Get positions with lowest entropy in their predictions."""
        self.printer.print(f"Computing position entropies for {num_positions} positions", increase_indent=True)

        if model_output is None:
            output = self.client.forward_and_sample(
                protein_tensor,
                sampling_configuration=SamplingConfig(
                    sequence=SamplingTrackConfig(temperature=self.temperature, topk_logprobs=5)
                )
            )
            self.cost += 1
            assert not isinstance(output, ESMProteinError), "Model forward_and_sample failed"
        else:
            output = model_output
            
        # Get entropy directly from output
        entropy = output.entropy.sequence
        if verbose: # Make entropy prints conditional
            self.printer.print(f"Raw entropies: {entropy}")
        
        # Mask entropy of already unmasked positions
        sequence_tensor = protein_tensor.sequence
        is_mask = sequence_tensor == self.MASK_TOKEN_ID # Use MASK_TOKEN_ID from Base class
        
        # Set entropy to inf for non-masked positions and start/end tokens
        entropy[~is_mask] = float('inf')
        entropy[0] = float('inf')
        entropy[-1] = float('inf')
        
        if verbose: # Make entropy prints conditional
            self.printer.print(f"Masked positions entropies: {entropy}")
        
        # Get positions with lowest entropy
        positions = []
        entropy_copy = entropy.clone()
        
        # Ensure we don't try to select more positions than available masks
        num_masked_positions = is_mask.sum().item()
        positions_to_select = min(num_positions, num_masked_positions)

        if positions_to_select == 0:
            self.printer.print("No masked positions left to select.", is_last=True, decrease_indent=True)
            return []

        for i in range(positions_to_select):
            next_pos = entropy_copy.argmin().item()
            positions.append(next_pos)
            entropy_copy[next_pos] = float('inf')  # Mark as processed
            
        self.printer.print(f"Selected positions: {positions}", is_last=True, decrease_indent=True)
        return positions
    
    def denoise(
        self,
        protein: ESMProtein,
        verbose: bool = True,
        max_decoding_steps: int = None,
    ) -> ESMProtein:
        """Denoise using entropy-based strategy."""
        self.printer.print("Starting entropy-based denoising process", increase_indent=True)
        
        # Use provided max_decoding_steps or fall back to self.num_decoding_steps
        if max_decoding_steps is None:
            max_decoding_steps = self.num_decoding_steps
        
        # Encode protein and add noise
        protein_tensor = self.client.encode(protein)
        assert not isinstance(protein_tensor, ESMProteinError), "Model encode failed"
        
        # Add noise by masking positions
        protein_tensor = self.add_noise(protein_tensor, self.noise_percentage, verbose=verbose) # Pass verbose
        
        # Calculate masked positions
        total_masked = self.get_number_of_masked_positions(protein_tensor)
        self.printer.print(f"Total masked positions: {total_masked}")

        if total_masked == 0:
            self.printer.print("No positions were masked, returning original protein.", is_last=True, decrease_indent=True)
            decoded_protein = self.client.decode(protein_tensor)
            assert not isinstance(decoded_protein, ESMProteinError), "Model decode failed"
            self.protein = decoded_protein
            return self.protein

        # Calculate how many positions to unmask per step
        max_steps = min(max_decoding_steps, self.num_decoding_steps)
        positions_per_step = self.calculate_positions_per_step(total_masked, max_steps)
        # Ensure at least one step is taken if there are masked positions
        actual_steps = max(1, min((total_masked + positions_per_step - 1) // positions_per_step, max_steps))
        
        self.printer.print(f"Positions per step: {positions_per_step}")
        self.printer.print(f"Actual number of steps: {actual_steps}")
        
        # Print initial sequence
        initial_decoded = self.client.decode(protein_tensor)
        assert not isinstance(initial_decoded, ESMProteinError), "Model decode failed"
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

            if positions_this_step == 0:
                self.printer.print("No positions left to denoise this step.", is_last=True, decrease_indent=True)
                continue # Should ideally not happen if actual_steps calculation is correct
                
            # Get model output once per step
            output = self.client.forward_and_sample(
                protein_tensor,
                sampling_configuration=SamplingConfig(
                    sequence=SamplingTrackConfig(temperature=self.temperature, topk_logprobs=5)
                )
            )
            self.cost += 1
            assert not isinstance(output, ESMProteinError), "Model forward_and_sample failed during step"
            
            # Get next positions using the same model output
            next_positions = self.get_next_positions(
                protein_tensor, positions_this_step, output, verbose=verbose # Pass verbose
            )

            if not next_positions:
                self.printer.print("No next positions identified, stopping early.", is_last=True, decrease_indent=True)
                break
            
            # Unmask using the same model output
            protein_tensor = self.unmask_positions(
                protein_tensor, next_positions, self.temperature, output
            )
            
            # Update remaining masked count
            remaining_masked -= len(next_positions)
            
            current_sequence_str = "(verbose mode off)"
            if verbose:
                after_unmask_decoded = self.client.decode(protein_tensor)
                assert not isinstance(after_unmask_decoded, ESMProteinError), "Model decode failed after unmasking"
                current_sequence_str = after_unmask_decoded.sequence
            
            self.printer.print(f"Current sequence: {current_sequence_str}", is_last=True, decrease_indent=True)

        self.printer.print("Denoising loop finished.", decrease_indent=True) # Back from steps

        # Check if all masks are removed, if not, run a final forward pass
        num_final_masked = self.get_number_of_masked_positions(protein_tensor)
        if num_final_masked > 0:
            self.printer.print(f"Performing final unmasking for {num_final_masked} remaining positions.")
            protein_tensor_output = self.client.forward_and_sample(
                protein_tensor,
                SamplingConfig(
                    sequence=SamplingTrackConfig(temperature=self.temperature)
                ),
            )
            self.cost += 1
            assert not isinstance(protein_tensor_output, ESMProteinError), "Final model forward_and_sample failed"
            protein_tensor = protein_tensor_output.protein_tensor
        else:
             self.printer.print(f"All positions unmasked.")

        decoded_protein = self.client.decode(protein_tensor)
        assert not isinstance(decoded_protein, ESMProteinError), "Final model decode failed"
        self.printer.print(f"Final denoised sequence: {decoded_protein.sequence}")
        self.printer.print(f"Total model calls: {self.cost}", is_last=True, decrease_indent=True) # Back from start
        self.protein = decoded_protein
        return decoded_protein

    def _extract_strategy_params(self) -> dict:
        params = super()._extract_strategy_params()
        params['strategy'] = 'entropy_based'
        return params

class MaxProbBasedDenoising(BaseDenoisingStrategy):
    """Denoising strategy that selects positions with highest probability for any token."""
    
    def get_next_positions(
        self,
        protein_tensor: ESMProteinTensor,
        num_positions: int,
        model_output = None,
        verbose: bool = True # Add verbose parameter
    ) -> list:
        """Get multiple positions to unmask based on maximum probability."""
        self.printer.print(f"Computing position probabilities for {num_positions} positions", increase_indent=True)
        
        if model_output is None:
            output = self.client.forward_and_sample(
                protein_tensor,
                sampling_configuration=SamplingConfig(
                    sequence=SamplingTrackConfig(temperature=1.0, topk_logprobs=5)
                )
            )
            self.cost += 1
            assert not isinstance(output, ESMProteinError)
        else:
            output = model_output
        
        # Get max probabilities from output
        if verbose: # Make topk_logprob print conditional
            print(output.topk_logprob.sequence)
        topk_logprobs = output.topk_logprob.sequence
        # Get the maximum probability for each position
        max_probs = torch.softmax(topk_logprobs, dim=-1).max(dim=-1).values
        if verbose: # Make probability prints conditional
            self.printer.print(f"Top probabilities at each position: {max_probs}")
        
        # Mask probabilities of already unmasked positions
        sequence_tensor = protein_tensor.sequence
        is_mask = sequence_tensor == self.MASK_TOKEN_ID # Use MASK_TOKEN_ID from Base class
        
        # Set probability to -inf for non-masked positions and start/end tokens
        max_probs[~is_mask] = -float('inf')  # Non-masked positions
        max_probs[0] = -float('inf')  # Start token
        max_probs[-1] = -float('inf')  # End token
        
        if verbose: # Make probability prints conditional
            self.printer.print(f"Masked positions probabilities: {max_probs}")
        
        # Get positions with highest probability
        positions = []
        probs_copy = max_probs.clone()
        
        for i in range(min(num_positions, is_mask.sum().item())):
            next_pos = probs_copy.argmax().item()
            positions.append(next_pos)
            probs_copy[next_pos] = -float('inf')  # Mark as processed
            
        self.printer.print(f"Selected positions: {positions}", is_last=True, decrease_indent=True)
        return positions
        
    def denoise(
        self,
        protein: ESMProtein,
        verbose: bool = True,
        max_decoding_steps: int = None,
    ) -> ESMProtein:
        """Denoise using max-probability strategy."""
        self.printer.print("Starting max-probability denoising process", increase_indent=True)
        
        # Use provided max_decoding_steps or fall back to self.num_decoding_steps
        if max_decoding_steps is None:
            max_decoding_steps = self.num_decoding_steps
        
        # Encode protein and add noise
        protein_tensor = self.client.encode(protein)
        assert not isinstance(protein_tensor, ESMProteinError)
        
        # Add noise by masking positions
        protein_tensor = self.add_noise(protein_tensor, self.noise_percentage, verbose=verbose) # Pass verbose
        
        # Calculate masked positions
        total_masked = self.get_number_of_masked_positions(protein_tensor)
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
                    sequence=SamplingTrackConfig(temperature=1.0, topk_logprobs=5)
                )
            )
            self.cost += 1
            assert not isinstance(output, ESMProteinError)
            
            # Get next positions using the same model output
            next_positions = self.get_next_positions(
                protein_tensor, positions_this_step, output, verbose=verbose # Pass verbose
            )
            
            # Unmask using the same model output
            protein_tensor = self.unmask_positions(
                protein_tensor, next_positions, self.temperature, output
            )
            
            # Update remaining masked count
            remaining_masked -= len(next_positions)
            
            if verbose:
                after_unmask_decoded = self.client.decode(protein_tensor)
                assert not isinstance(after_unmask_decoded, ESMProteinError)
                self.printer.print(f"Current sequence: {after_unmask_decoded.sequence}", 
                                 is_last=True, decrease_indent=True)
            else:
                # *** Ensure indent decreases even if not printing the sequence ***
                self.printer.decrease_indent() # This balances the increase_indent at the start of the loop step
        
        self.printer.print("Denoising loop finished.", decrease_indent=True) # Back from steps
        
        # Final prediction
        protein_tensor_output = self.client.forward_and_sample(
            protein_tensor,
            SamplingConfig(
                sequence=SamplingTrackConfig(temperature=self.temperature)
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
    
class SimulatedAnnealingDenoising(BaseDenoisingStrategy):
    """Denoising strategy that uses simulated annealing for temperature control."""
    
    def __init__(
        self,
        client: ESM3InferenceClient,
        noise_percentage: float = 50.0,
        num_decoding_steps: int = 20,
        base_temperature: float = 1.0,  # Maximum temperature at the start
        schedule_type: str = 'linear', # 'linear' or 'cosine'
    ):
        # Initialize with base_temperature, but the actual temperature used will vary
        super().__init__(client, noise_percentage, num_decoding_steps, temperature=0.0) # Base class temp not used directly
        self.base_temperature = base_temperature
        if schedule_type not in ['linear', 'cosine']:
            raise ValueError("schedule_type must be 'linear' or 'cosine'")
        self.schedule_type = schedule_type
        self.initial_mask_fraction: Optional[float] = None # Store initial mask fraction

    def get_temperature(self, protein_tensor: ESMProteinTensor, verbose: bool = True) -> float: # Add verbose parameter
        """Get temperature based on number of masked positions and schedule type."""
        num_masked = self.get_number_of_masked_positions(protein_tensor)
        total_positions = len(protein_tensor.sequence) - 2  # Exclude start/end tokens

        if total_positions <= 0 or self.initial_mask_fraction is None or self.initial_mask_fraction == 0:
            # Handle edge cases: no positions, denoising not started, or no initial masks
            if verbose: # Make print conditional
                self.printer.print(f"Calculated temperature: 0.0 (Edge case: total_positions={total_positions}, initial_mask_fraction={self.initial_mask_fraction})")
            return 0.0

        current_mask_fraction = num_masked / total_positions

        if self.schedule_type == 'linear':
            # Scale linearly from base_temperature to 0 based on remaining mask fraction relative to initial
            temperature = self.base_temperature * (current_mask_fraction / self.initial_mask_fraction)
            # Clamp temperature between 0 and base_temperature
            temperature = max(0.0, min(temperature, self.base_temperature))
            if verbose: # Make print conditional
                self.printer.print(f"Calculated temperature (Linear): {temperature:.4f} (Current Mask: {current_mask_fraction:.2f}, Initial Mask: {self.initial_mask_fraction:.2f})")

        elif self.schedule_type == 'cosine':
            # Calculate progress (0 at start, 1 at end)
            progress = 1.0 - (current_mask_fraction / self.initial_mask_fraction)
            # Clamp progress between 0 and 1 to handle potential float inaccuracies
            progress = max(0.0, min(progress, 1.0))
            # Cosine schedule: T_max * 0.5 * (1 + cos(pi * progress))
            temperature = self.base_temperature * 0.5 * (1.0 + math.cos(math.pi * progress))
            # Clamp temperature between 0 and base_temperature
            temperature = max(0.0, min(temperature, self.base_temperature))
            if verbose: # Make print conditional
                self.printer.print(f"Calculated temperature (Cosine): {temperature:.4f} (Progress: {progress:.2f}, Initial Mask: {self.initial_mask_fraction:.2f})")
        else:
            # Should not happen due to check in __init__
            temperature = 0.0
            if verbose: # Make print conditional
                self.printer.print(f"Calculated temperature: 0.0 (Unknown schedule type)")


        return temperature

    def get_next_positions(
        self,
        protein_tensor: ESMProteinTensor,
        num_positions: int,
        model_output = None,
        verbose: bool = True # Add verbose parameter
    ) -> List[int]:
        """Get positions with lowest entropy (highest uncertainty) for annealing."""
        # This should use the same logic as EntropyBasedDenoising: select lowest entropy
        # Note: get_temperature is now called within the denoise loop before this method,
        # so we don't call it here anymore to avoid redundant calculations/prints.
        # The temperature used for the forward pass is handled in the denoise loop.

        self.printer.print(f"Computing position entropies for {num_positions} positions (Simulated Annealing)", increase_indent=True)

        # We assume model_output is always provided by the denoise loop,
        # which already ran forward_and_sample with the correct temperature.
        if model_output is None:
             # This path should ideally not be taken if called from the standard denoise loop.
             # If it needs to be supported, the temperature logic would need adjustment here.
             self.printer.print("Warning: model_output not provided to get_next_positions. Recalculating with potentially incorrect temperature.", is_warning=True)
             current_temperature = self.get_temperature(protein_tensor, verbose=verbose) # Get temp if needed
             output = self.client.forward_and_sample(
                 protein_tensor,
                 sampling_configuration=SamplingConfig(
                     sequence=SamplingTrackConfig(temperature=current_temperature, topk_logprobs=5)
                 )
             )
             self.cost += 1
             assert not isinstance(output, ESMProteinError), "Model forward_and_sample failed in get_next_positions"
        else:
            output = model_output

        # Get entropy directly from output
        entropy = output.entropy.sequence
        if verbose: # Make entropy prints conditional
            self.printer.print(f"Raw entropies: {entropy}")

        # Mask entropy of already unmasked positions
        sequence_tensor = protein_tensor.sequence
        is_mask = sequence_tensor == self.MASK_TOKEN_ID # Use MASK_TOKEN_ID from Base class

        # Set entropy to inf for non-masked positions and start/end tokens
        entropy[~is_mask] = float('inf')
        entropy[0] = float('inf')
        entropy[-1] = float('inf')

        if verbose: # Make entropy prints conditional
            self.printer.print(f"Masked positions entropies: {entropy}")

        # Get positions with lowest entropy
        positions = []
        entropy_copy = entropy.clone()

        # Ensure we don't try to select more positions than available masks
        num_masked_positions = is_mask.sum().item()
        positions_to_select = min(num_positions, num_masked_positions)

        if positions_to_select == 0:
            self.printer.print("No masked positions left to select.", is_last=True, decrease_indent=True)
            return []

        for i in range(positions_to_select):
            next_pos = entropy_copy.argmin().item()
            positions.append(next_pos)
            entropy_copy[next_pos] = float('inf')  # Mark as processed

        self.printer.print(f"Selected positions: {positions}", is_last=True, decrease_indent=True)
        return positions

    def denoise(
        self,
        protein: ESMProtein,
        verbose: bool = True,
        max_decoding_steps: int = None,
    ) -> ESMProtein:
        """Denoise using simulated annealing strategy."""
        self.printer.print(f"Starting simulated annealing denoising process (Schedule: {self.schedule_type})", increase_indent=True)
        self.cost = 0 # Reset cost counter
        self.initial_mask_fraction = None # Reset initial mask fraction

        # Use provided max_decoding_steps or fall back to self.num_decoding_steps
        if max_decoding_steps is None:
            max_decoding_steps = self.num_decoding_steps

        # Encode protein and add noise
        protein_tensor = self.client.encode(protein)
        assert not isinstance(protein_tensor, ESMProteinError), "Model encode failed"

        # Add noise by masking positions
        protein_tensor = self.add_noise(protein_tensor, self.noise_percentage, verbose=verbose) # Pass verbose

        # Calculate masked positions
        total_masked = self.get_number_of_masked_positions(protein_tensor)
        self.printer.print(f"Total masked positions: {total_masked}")

        # Store initial mask fraction
        total_positions = len(protein_tensor.sequence) - 2
        if total_positions > 0:
            self.initial_mask_fraction = total_masked / total_positions
            if verbose: # Only print if verbose
                self.printer.print(f"Initial mask fraction: {self.initial_mask_fraction:.4f}")
        else:
            self.initial_mask_fraction = 0.0 # Or None, handled in get_temperature
            if verbose: # Only print if verbose
                self.printer.print("Initial mask fraction: N/A (sequence too short)")


        if total_masked == 0:
            self.printer.print("No positions were masked, returning original protein.", is_last=True, decrease_indent=True)
            decoded_protein = self.client.decode(protein_tensor)
            assert not isinstance(decoded_protein, ESMProteinError), "Model decode failed"
            self.protein = decoded_protein
            return self.protein

        # Calculate how many positions to unmask per step
        max_steps = min(max_decoding_steps, self.num_decoding_steps)
        positions_per_step = self.calculate_positions_per_step(total_masked, max_steps)
        # Ensure at least one step is taken if there are masked positions
        actual_steps = max(1, min((total_masked + positions_per_step - 1) // positions_per_step, max_steps))

        self.printer.print(f"Positions per step: {positions_per_step}")
        self.printer.print(f"Actual number of steps: {actual_steps}")

        # Print initial sequence
        initial_decoded = self.client.decode(protein_tensor)
        assert not isinstance(initial_decoded, ESMProteinError), "Model decode failed"
        if verbose: # Only print if verbose
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

            if positions_this_step == 0:
                self.printer.print("No positions left to denoise this step.", is_last=True, decrease_indent=True)
                continue

            # Get current temperature for this step, passing verbose flag
            current_temperature = self.get_temperature(protein_tensor, verbose=verbose)

            # Get model output once per step, using current annealing temperature
            output = self.client.forward_and_sample(
                protein_tensor,
                sampling_configuration=SamplingConfig(
                    sequence=SamplingTrackConfig(temperature=current_temperature, topk_logprobs=5)
                )
            )
            self.cost += 1
            assert not isinstance(output, ESMProteinError), "Model forward_and_sample failed during step"

            # Get next positions using the same model output (based on lowest entropy)
            next_positions = self.get_next_positions(
                protein_tensor, positions_this_step, output, verbose=verbose # Pass verbose
            )

            if not next_positions:
                self.printer.print("No next positions identified, stopping early.", is_last=True, decrease_indent=True)
                break

            # Unmask using the same model output and the *current* temperature
            protein_tensor = self.unmask_positions(
                protein_tensor, next_positions, current_temperature, output
            )

            # Update remaining masked count
            remaining_masked -= len(next_positions)

            current_sequence_str = "(verbose mode off)"
            if verbose:
                after_unmask_decoded = self.client.decode(protein_tensor)
                assert not isinstance(after_unmask_decoded, ESMProteinError), "Model decode failed after unmasking"
                current_sequence_str = after_unmask_decoded.sequence
                self.printer.print(f"Current sequence: {current_sequence_str}", is_last=True, decrease_indent=True)
            else:
                # Decrease indent even if not printing the sequence when verbose is off
                self.printer.decrease_indent() # Ensure indent decreases


        self.printer.print("Denoising loop finished.", decrease_indent=True) # Back from steps

        # Check if all masks are removed, if not, run a final forward pass with T=0
        num_final_masked = self.get_number_of_masked_positions(protein_tensor)
        if num_final_masked > 0:
            if verbose: # Only print if verbose
                self.printer.print(f"Performing final unmasking for {num_final_masked} remaining positions (T=0).")
            protein_tensor_output = self.client.forward_and_sample(
                protein_tensor,
                SamplingConfig(
                    sequence=SamplingTrackConfig(temperature=0.0) # Use T=0 for final greedy selection
                ),
            )
            self.cost += 1
            assert not isinstance(protein_tensor_output, ESMProteinError), "Final model forward_and_sample failed"
            protein_tensor = protein_tensor_output.protein_tensor
        else:
             if verbose: # Only print if verbose
                 self.printer.print(f"All positions unmasked.")

        decoded_protein = self.client.decode(protein_tensor)
        assert not isinstance(decoded_protein, ESMProteinError), "Final model decode failed"
        self.printer.print(f"Final denoised sequence: {decoded_protein.sequence}")
        self.printer.print(f"Total model calls: {self.cost}", is_last=True, decrease_indent=True) # Back from start
        self.protein = decoded_protein
        return decoded_protein
    
    def _extract_strategy_params(self) -> dict:
        params = super()._extract_strategy_params()
        params['strategy'] = 'simulated_annealing'
        params['base_temperature'] = self.base_temperature
        params['schedule_type'] = self.schedule_type # Add schedule type
        return params

class OneShotDenoising(BaseDenoisingStrategy):
    """Denoising strategy that unmasks all positions in a single step."""

    def __init__(
        self,
        client: ESM3InferenceClient,
        noise_percentage: float = 50.0,
        temperature: float = 0.0,
    ):
        # num_decoding_steps is irrelevant here, set to 1 conceptually
        super().__init__(client, noise_percentage, num_decoding_steps=1, temperature=temperature)

    def get_next_positions(
        self,
        protein_tensor: ESMProteinTensor,
        num_positions: int,
        model_output = None,
        verbose: bool = True # Add verbose parameter (though not used)
    ) -> List[int]:
        """Not used in OneShotDenoising."""
        raise NotImplementedError("get_next_positions is not applicable for OneShotDenoising")

    def denoise(
        self,
        protein: ESMProtein,
        verbose: bool = True,
        max_decoding_steps: int = None, # Ignored, always 1 step
    ) -> ESMProtein:
        """Denoise using a single forward pass."""
        self.printer.print("Starting one-shot denoising process", increase_indent=True)
        self.cost = 0 # Reset cost counter

        # Encode protein
        protein_tensor = self.client.encode(protein)
        assert not isinstance(protein_tensor, ESMProteinError), "Model encode failed"

        # Add noise by masking positions
        protein_tensor = self.add_noise(protein_tensor, self.noise_percentage, verbose=verbose) # Pass verbose

        # Calculate masked positions
        total_masked = self.get_number_of_masked_positions(protein_tensor)
        self.printer.print(f"Total masked positions: {total_masked}")

        if total_masked == 0:
            self.printer.print("No positions were masked, returning original protein.", is_last=True, decrease_indent=True)
            decoded_protein = self.client.decode(protein_tensor)
            assert not isinstance(decoded_protein, ESMProteinError), "Model decode failed"
            self.protein = decoded_protein
            return self.protein

        # Print initial sequence
        initial_decoded = self.client.decode(protein_tensor)
        assert not isinstance(initial_decoded, ESMProteinError), "Model decode failed"
        self.printer.print(f"Initial sequence: {initial_decoded.sequence}")

        # Perform a single forward pass to unmask everything
        self.printer.print(f"Performing single forward pass to unmask {total_masked} positions with T={self.temperature}.")
        protein_tensor_output = self.client.forward_and_sample(
            protein_tensor,
            SamplingConfig(
                sequence=SamplingTrackConfig(temperature=self.temperature)
            ),
        )
        self.cost += 1
        assert not isinstance(protein_tensor_output, ESMProteinError), "One-shot model forward_and_sample failed"

        # The result is directly in the output tensor
        protein_tensor = protein_tensor_output.protein_tensor

        # Decode the final result
        decoded_protein = self.client.decode(protein_tensor)
        assert not isinstance(decoded_protein, ESMProteinError), "Final model decode failed"
        self.printer.print(f"Final denoised sequence: {decoded_protein.sequence}")
        self.printer.print(f"Total model calls: {self.cost}", is_last=True, decrease_indent=True) # Back from start
        self.protein = decoded_protein
        return decoded_protein

    def _extract_strategy_params(self) -> dict:
        params = super()._extract_strategy_params()
        params['strategy'] = 'one_shot'
        # Remove num_decoding_steps as it's always 1
        params.pop('num_decoding_steps', None)
        return params