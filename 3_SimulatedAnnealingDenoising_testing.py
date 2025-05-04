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

# Import classes from denoising_strategies.py
from denoising_strategies import Tee, PrintFormatter, SimulatedAnnealingDenoising

# Initialize the ESM3InferenceClient
from esm.sdk import client
token = os.getenv("ESM_FORGE_API_KEY")
model = client(model="esm3-large-2024-03", url="https://forge.evolutionaryscale.ai", token=token)

# --- Configuration ---
TEST_SEQUENCE = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"  # A longer test sequence for SA
NOISE_PERCENTAGE = 50.0  # Mask 50% initially
NUM_DECODING_STEPS = 5  # Number of steps to unmask
BASE_TEMPERATURE = 1.0
# --- End Configuration ---

# Create a dummy protein
protein = ESMProtein(sequence=TEST_SEQUENCE)
print(f"Original Protein: {protein.sequence}\n")

# Create and test the denoiser
print("\n=== Testing SimulatedAnnealingDenoising ===")
sa_denoiser = SimulatedAnnealingDenoising(
    client=model,
    noise_percentage=NOISE_PERCENTAGE,
    num_decoding_steps=NUM_DECODING_STEPS,
    base_temperature=BASE_TEMPERATURE
)

# Test the denoising process
sa_denoiser.denoise(
    protein=protein,
    verbose=True
)
denoised_protein, cost = sa_denoiser.return_generation()

print(f"\nOriginal sequence: {protein.sequence}")
print(f"Denoised sequence: {denoised_protein.sequence}")
print(f"Final cost: {cost}")