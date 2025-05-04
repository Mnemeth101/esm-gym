from .base import BaseDenoisingStrategy
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Callable
from esm.sdk.api import ESM3InferenceClient, ESMProteinTensor, ESMProtein

class RewardGuidedBaseDenoisingStrategy(BaseDenoisingStrategy):
    """Base class for reward-guided denoising strategies."""
    
    @classmethod
    def get_required_params(cls):
        params = super().get_required_params()
        params.update({
            'reward_function',
            'lookahead_steps',
        })
        return params
    
    def __init__(
        self,
        client: ESM3InferenceClient,
        noise_percentage: float = 50.0,
        num_decoding_steps: int = 20,
        temperature: float = 0.0,
        reward_function: Callable[[ESMProtein], float] = None,
        lookahead_steps: int = 1,
    ):
        super().__init__(client, noise_percentage, num_decoding_steps, temperature)
        self.reward_function = reward_function
        self.lookahead_steps = lookahead_steps
    
    def predict_complete_protein(
        self,
        protein_tensor: ESMProteinTensor,
        lookahead_steps: Optional[int] = None
    ) -> ESMProtein:
        """Predict the complete protein sequence."""
        if lookahead_steps is None:
            lookahead_steps = self.lookahead_steps
            
        # Get initial model output
        output = self.client.forward(protein_tensor)
        self.cost += 1
        
        # Create a copy of the protein tensor
        protein_tensor = attr.evolve(protein_tensor)
        sequence_tensor = protein_tensor.sequence.clone()
        
        # Get masked positions
        is_mask = sequence_tensor == 32  # MASK_TOKEN_ID
        masked_positions = torch.where(is_mask)[0]
        
        if len(masked_positions) == 0:
            return ESMProtein(protein_tensor)
            
        # For each masked position
        for position in masked_positions:
            # Get logits for this position
            logits = output.logits.sequence[position]
            
            # Sample from logits
            probs = F.softmax(logits / self.temperature, dim=-1)
            token_id = torch.multinomial(probs, 1).item()
            
            # Update sequence
            sequence_tensor[position] = token_id
            
            # If we need to look ahead
            if lookahead_steps > 0:
                # Create new protein tensor with updated sequence
                new_protein_tensor = attr.evolve(protein_tensor)
                new_protein_tensor.sequence = sequence_tensor
                
                # Recursively predict with one less lookahead step
                return self.predict_complete_protein(
                    new_protein_tensor,
                    lookahead_steps - 1
                )
        
        # If no lookahead needed, return final protein
        protein_tensor.sequence = sequence_tensor
        return ESMProtein(protein_tensor)
    
    def evaluate_position_rewards(
        self,
        protein_tensor: ESMProteinTensor,
        candidate_positions: List[int],
        model_output = None
    ) -> Dict[int, float]:
        """Evaluate rewards for each candidate position."""
        if model_output is None:
            output = self.client.forward(protein_tensor)
            self.cost += 1
        else:
            output = model_output
            
        rewards = {}
        
        # For each candidate position
        for position in candidate_positions:
            # Get logits for this position
            logits = output.logits.sequence[position]
            
            # Sample from logits
            probs = F.softmax(logits / self.temperature, dim=-1)
            token_id = torch.multinomial(probs, 1).item()
            
            # Create new protein tensor with predicted token
            new_protein_tensor = attr.evolve(protein_tensor)
            sequence_tensor = new_protein_tensor.sequence.clone()
            sequence_tensor[position] = token_id
            new_protein_tensor.sequence = sequence_tensor
            
            # Predict complete protein
            predicted_protein = self.predict_complete_protein(new_protein_tensor)
            
            # Calculate reward
            reward = self.reward_function(predicted_protein)
            rewards[position] = reward
            
        return rewards
    
    def get_next_positions(
        self,
        protein_tensor: ESMProteinTensor,
        num_positions: int,
        model_output = None
    ) -> List[int]:
        """Get positions with highest reward in their predictions."""
        # Get masked positions
        sequence_tensor = protein_tensor.sequence
        is_mask = sequence_tensor == 32  # MASK_TOKEN_ID
        masked_positions = torch.where(is_mask)[0]
        
        if len(masked_positions) == 0:
            return []
            
        # Evaluate rewards for all masked positions
        rewards = self.evaluate_position_rewards(
            protein_tensor,
            masked_positions.tolist(),
            model_output
        )
        
        # Sort positions by reward
        sorted_positions = sorted(
            rewards.keys(),
            key=lambda p: rewards[p],
            reverse=True
        )
        
        # Return top positions
        return sorted_positions[:num_positions]
    
    def _extract_strategy_params(self) -> dict:
        params = super()._extract_strategy_params()
        params.update({
            'strategy': 'reward_guided',
            'lookahead_steps': self.lookahead_steps,
        })
        return params

class EntropyBasedRewardGuidedDenoising(RewardGuidedBaseDenoisingStrategy):
    """Reward-guided denoising that uses entropy for initial position selection."""
    
    @classmethod
    def get_required_params(cls):
        return super().get_required_params()
    
    def get_next_positions(
        self,
        protein_tensor: ESMProteinTensor,
        num_positions: int,
        model_output = None
    ) -> List[int]:
        """Get positions with highest entropy first, then evaluate rewards."""
        if model_output is None:
            output = self.client.forward(protein_tensor)
            self.cost += 1
        else:
            output = model_output
            
        # Get masked positions
        sequence_tensor = protein_tensor.sequence
        is_mask = sequence_tensor == 32  # MASK_TOKEN_ID
        masked_positions = torch.where(is_mask)[0]
        
        if len(masked_positions) == 0:
            return []
            
        # Calculate entropy for each position
        logits = output.logits.sequence[masked_positions]
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        # Get top positions by entropy
        _, indices = torch.topk(entropy, min(num_positions * 2, len(masked_positions)))
        candidate_positions = masked_positions[indices].tolist()
        
        # Evaluate rewards for candidate positions
        rewards = self.evaluate_position_rewards(
            protein_tensor,
            candidate_positions,
            model_output
        )
        
        # Sort by reward
        sorted_positions = sorted(
            rewards.keys(),
            key=lambda p: rewards[p],
            reverse=True
        )
        
        # Return top positions
        return sorted_positions[:num_positions]

class MaxProbBasedRewardGuidedDenoising(RewardGuidedBaseDenoisingStrategy):
    """Reward-guided denoising that uses max probability for initial position selection."""
    
    @classmethod
    def get_required_params(cls):
        return super().get_required_params()
    
    def get_next_positions(
        self,
        protein_tensor: ESMProteinTensor,
        num_positions: int,
        model_output = None
    ) -> List[int]:
        """Get positions with highest max probability first, then evaluate rewards."""
        if model_output is None:
            output = self.client.forward(protein_tensor)
            self.cost += 1
        else:
            output = model_output
            
        # Get masked positions
        sequence_tensor = protein_tensor.sequence
        is_mask = sequence_tensor == 32  # MASK_TOKEN_ID
        masked_positions = torch.where(is_mask)[0]
        
        if len(masked_positions) == 0:
            return []
            
        # Calculate max probability for each position
        logits = output.logits.sequence[masked_positions]
        probs = F.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        
        # Get top positions by max probability
        _, indices = torch.topk(max_probs, min(num_positions * 2, len(masked_positions)))
        candidate_positions = masked_positions[indices].tolist()
        
        # Evaluate rewards for candidate positions
        rewards = self.evaluate_position_rewards(
            protein_tensor,
            candidate_positions,
            model_output
        )
        
        # Sort by reward
        sorted_positions = sorted(
            rewards.keys(),
            key=lambda p: rewards[p],
            reverse=True
        )
        
        # Return top positions
        return sorted_positions[:num_positions]

class SimulatedAnnealingRewardGuidedDenoising(RewardGuidedBaseDenoisingStrategy):
    """Reward-guided denoising that uses simulated annealing for temperature control."""
    
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
        base_temperature: float = 1.0,
        reward_function: Callable[[ESMProtein], float] = None,
        lookahead_steps: int = 1,
    ):
        # Call parent constructor with temperature=0.0 (will be overridden by get_temperature)
        super().__init__(
            client,
            noise_percentage,
            num_decoding_steps,
            temperature=0.0,
            reward_function=reward_function,
            lookahead_steps=lookahead_steps
        )
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
    
    def _extract_strategy_params(self) -> dict:
        params = super()._extract_strategy_params()
        params['base_temperature'] = self.base_temperature
        return params 