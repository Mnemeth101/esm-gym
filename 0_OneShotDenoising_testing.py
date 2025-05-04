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
from denoising_strategies import Tee, PrintFormatter, OneShotDenoising # Changed import

# Initialize the ESM3InferenceClient
from esm.sdk import client
token = os.getenv("ESM_FORGE_API_KEY")
model = client(model="esm3-large-2024-03", url="https://forge.evolutionaryscale.ai", token=token)
# --- Configuration ---
TEST_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY" # Longer sequence for testing
NOISE_PERCENTAGE = 50.0  # Mask 50% initially
TEMPERATURE = 0.0
# NUM_DECODING_STEPS is not applicable for OneShotDenoising
TRACK = "sequence"
# --- End Configuration ---

# Create a dummy protein
protein = ESMProtein(sequence=TEST_SEQUENCE)
print(f"Original Protein: {protein.sequence}\n")

# Create and test the denoiser
print("\n=== Testing OneShotDenoising ===") # Changed print statement
one_shot_denoiser = OneShotDenoising( # Changed class instantiation
    client=model,
    noise_percentage=NOISE_PERCENTAGE,
    temperature=TEMPERATURE
    # num_decoding_steps is not needed
)

# Test the denoising process
denoised_protein_oneshot = one_shot_denoiser.denoise( # Changed variable name
    protein=protein,
    verbose=True
)
