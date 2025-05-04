from .base import BaseDenoisingStrategy
import torch
import torch.nn.functional as F
from typing import List, Optional
from esm.sdk.api import ESM3InferenceClient, ESMProteinTensor

class EntropyBasedDenoising(BaseDenoisingStrategy):
    """Denoising strategy that selects positions based on entropy of predictions."""
    
    @classmethod
    def get_required_params(cls):
        return super().get_required_params()
    
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
        """Get positions with highest entropy in their predictions."""
        if model_output is None:
            output = self.client.forward(protein_tensor)
            self.cost += 1
        else:
            output = model_output
            
        # Get logits for masked positions
        sequence_tensor = protein_tensor.sequence
        is_mask = sequence_tensor == 32  # MASK_TOKEN_ID
        masked_positions = torch.where(is_mask)[0]
        
        if len(masked_positions) == 0:
            return []
            
        # Get logits for masked positions
        logits = output.logits.sequence[masked_positions]
        
        # Calculate entropy for each position
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        # Get positions with highest entropy
        _, indices = torch.topk(entropy, min(num_positions, len(masked_positions)))
        selected_positions = masked_positions[indices].tolist()
        
        return selected_positions
    
    def _extract_strategy_params(self) -> dict:
        params = super()._extract_strategy_params()
        params['strategy'] = 'entropy_based'
        return params

class MaxProbBasedDenoising(BaseDenoisingStrategy):
    """Denoising strategy that selects positions based on maximum probability."""
    
    @classmethod
    def get_required_params(cls):
        return super().get_required_params()
    
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
        """Get positions with highest maximum probability in their predictions."""
        if model_output is None:
            output = self.client.forward(protein_tensor)
            self.cost += 1
        else:
            output = model_output
            
        # Get logits for masked positions
        sequence_tensor = protein_tensor.sequence
        is_mask = sequence_tensor == 32  # MASK_TOKEN_ID
        masked_positions = torch.where(is_mask)[0]
        
        if len(masked_positions) == 0:
            return []
            
        # Get logits for masked positions
        logits = output.logits.sequence[masked_positions]
        
        # Calculate max probability for each position
        probs = F.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        
        # Get positions with highest max probability
        _, indices = torch.topk(max_probs, min(num_positions, len(masked_positions)))
        selected_positions = masked_positions[indices].tolist()
        
        return selected_positions
    
    def _extract_strategy_params(self) -> dict:
        params = super()._extract_strategy_params()
        params['strategy'] = 'max_prob_based'
        return params

class SimulatedAnnealingDenoising(BaseDenoisingStrategy):
    """Denoising strategy that uses simulated annealing for temperature control."""
    
    @classmethod
    def get_required_params(cls):
        params = super().get_required_params()
        params.add('base_temperature')
        return params
    
    def __init__(
        self,
        client: ESM3InferenceClient,
        noise_percentage: float = 50.0,
        num_decoding_steps: int = 20,
        base_temperature: float = 1.0,  # Maximum temperature when fully masked
    ):
        super().__init__(client, noise_percentage, num_decoding_steps, temperature=0.0)
        self.base_temperature = base_temperature
    
    def get_temperature(self, protein_tensor: ESMProteinTensor) -> float:
        """Get temperature based on number of masked positions."""
        num_masked = self.get_number_of_masked_positions(protein_tensor)
        total_positions = len(protein_tensor.sequence) - 2  # Exclude start/end tokens
        
        if total_positions == 0:
            return 0.0
            
        # Temperature decreases linearly with unmasked positions
        progress = 1.0 - (num_masked / total_positions)
        temperature = self.base_temperature * (1.0 - progress)
        
        return temperature
    
    def get_next_positions(
        self,
        protein_tensor: ESMProteinTensor,
        num_positions: int,
        model_output = None
    ) -> List[int]:
        """Get positions with highest entropy in their predictions."""
        if model_output is None:
            output = self.client.forward(protein_tensor)
            self.cost += 1
        else:
            output = model_output
            
        # Get logits for masked positions
        sequence_tensor = protein_tensor.sequence
        is_mask = sequence_tensor == 32  # MASK_TOKEN_ID
        masked_positions = torch.where(is_mask)[0]
        
        if len(masked_positions) == 0:
            return []
            
        # Get logits for masked positions
        logits = output.logits.sequence[masked_positions]
        
        # Calculate entropy for each position
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        # Get positions with highest entropy
        _, indices = torch.topk(entropy, min(num_positions, len(masked_positions)))
        selected_positions = masked_positions[indices].tolist()
        
        return selected_positions
    
    def _extract_strategy_params(self) -> dict:
        params = super()._extract_strategy_params()
        params['strategy'] = 'simulated_annealing'
        params['base_temperature'] = self.base_temperature
        return params 