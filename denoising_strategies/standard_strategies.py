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
        model_output = None
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
        self.printer.print(f"Raw entropies: {entropy}")
        
        # Mask entropy of already unmasked positions
        sequence_tensor = protein_tensor.sequence
        is_mask = sequence_tensor == self.MASK_TOKEN_ID # Use MASK_TOKEN_ID from Base class
        
        # Set entropy to inf for non-masked positions and start/end tokens
        entropy[~is_mask] = float('inf')
        entropy[0] = float('inf')
        entropy[-1] = float('inf')
        
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
        protein_tensor = self.add_noise(protein_tensor, self.noise_percentage)
        
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
                protein_tensor, positions_this_step, output
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
        model_output = None
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
        print(output.topk_logprob.sequence)
        topk_logprobs = output.topk_logprob.sequence
        # Get the maximum probability for each position
        max_probs = torch.softmax(topk_logprobs, dim=-1).max(dim=-1).values
        self.printer.print(f"Top probabilities at each position: {max_probs}")
        
        # Mask probabilities of already unmasked positions
        sequence_tensor = protein_tensor.sequence
        is_mask = sequence_tensor == self.MASK_TOKEN_ID # Use MASK_TOKEN_ID from Base class
        
        # Set probability to -inf for non-masked positions and start/end tokens
        max_probs[~is_mask] = -float('inf')  # Non-masked positions
        max_probs[0] = -float('inf')  # Start token
        max_probs[-1] = -float('inf')  # End token
        
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
        protein_tensor = self.add_noise(protein_tensor, self.noise_percentage)
        
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
                protein_tensor, positions_this_step, output
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
        base_temperature: float = 1.0,  # Maximum temperature when fully masked
    ):
        # Initialize with base_temperature, but the actual temperature used will vary
        super().__init__(client, noise_percentage, num_decoding_steps, temperature=0.0)
        self.base_temperature = base_temperature
    
    def get_temperature(self, protein_tensor: ESMProteinTensor) -> float:
        """Get temperature based on number of masked positions."""
        num_masked = self.get_number_of_masked_positions(protein_tensor)
        # Ensure total_positions is calculated based on the tensor length
        total_positions = len(protein_tensor.sequence) - 2  # Exclude start/end tokens
        
        if total_positions <= 0: # Avoid division by zero if sequence is too short
            return 0.0
            
        # Temperature decreases linearly as positions get unmasked
        mask_fraction = num_masked / total_positions
        temperature = self.base_temperature * mask_fraction # Temperature proportional to mask fraction
        
        self.printer.print(f"Calculated temperature: {temperature:.4f} (Masked: {num_masked}/{total_positions})")
        return temperature
    
    def get_next_positions(
        self,
        protein_tensor: ESMProteinTensor,
        num_positions: int,
        model_output = None
    ) -> List[int]:
        """Get positions with lowest entropy (highest uncertainty) for annealing."""
        # This should use the same logic as EntropyBasedDenoising: select lowest entropy
        self.printer.print(f"Computing position entropies for {num_positions} positions (Simulated Annealing)", increase_indent=True)

        current_temperature = self.get_temperature(protein_tensor)

        if model_output is None:
            # Use current_temperature for sampling if getting new output
            output = self.client.forward_and_sample(
                protein_tensor,
                sampling_configuration=SamplingConfig(
                    sequence=SamplingTrackConfig(temperature=current_temperature, topk_logprobs=5)
                )
            )
            self.cost += 1
            assert not isinstance(output, ESMProteinError), "Model forward_and_sample failed"
        else:
            output = model_output
            
        # Get entropy directly from output
        entropy = output.entropy.sequence
        self.printer.print(f"Raw entropies: {entropy}")
        
        # Mask entropy of already unmasked positions
        sequence_tensor = protein_tensor.sequence
        is_mask = sequence_tensor == self.MASK_TOKEN_ID # Use MASK_TOKEN_ID from Base class
        
        # Set entropy to inf for non-masked positions and start/end tokens
        entropy[~is_mask] = float('inf')
        entropy[0] = float('inf')
        entropy[-1] = float('inf')
        
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
        self.printer.print("Starting simulated annealing denoising process", increase_indent=True)
        
        # Use provided max_decoding_steps or fall back to self.num_decoding_steps
        if max_decoding_steps is None:
            max_decoding_steps = self.num_decoding_steps
        
        # Encode protein and add noise
        protein_tensor = self.client.encode(protein)
        assert not isinstance(protein_tensor, ESMProteinError), "Model encode failed"
        
        # Add noise by masking positions
        protein_tensor = self.add_noise(protein_tensor, self.noise_percentage)
        
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
                continue

            # Get current temperature for this step
            current_temperature = self.get_temperature(protein_tensor)
                
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
                protein_tensor, positions_this_step, output
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

        self.printer.print("Denoising loop finished.", decrease_indent=True) # Back from steps
        
        # Check if all masks are removed, if not, run a final forward pass with T=0
        num_final_masked = self.get_number_of_masked_positions(protein_tensor)
        if num_final_masked > 0:
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
        return params 