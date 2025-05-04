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
from denoising_strategies import Tee, PrintFormatter, MaxProbBasedDenoising

# Initialize the ESM3InferenceClient
from esm.sdk import client
token = os.getenv("ESM_FORGE_API_KEY")
model = client(model="esm3-open", url="https://forge.evolutionaryscale.ai", token=token)

# --- Configuration ---
TEST_SEQUENCE = "ACDE"
NOISE_PERCENTAGE = 50.0  # Mask 50% initially (2 positions for length 4)
NUM_DECODING_STEPS = 2  # Number of steps to unmask
TEMPERATURE = 0.0
TRACK = "sequence"
# --- End Configuration ---

# Create a dummy protein
protein = ESMProtein(sequence=TEST_SEQUENCE)
print(f"Original Protein: {protein.sequence}\n")

# Create and test the denoiser
print("\n=== Testing MaxProbBasedDenoising ===")
max_prob_denoiser = MaxProbBasedDenoising(
    client=model,
    noise_percentage=NOISE_PERCENTAGE,
    num_decoding_steps=NUM_DECODING_STEPS,
    temperature=TEMPERATURE
)

# Test the denoising process
denoised_protein_maxprob = max_prob_denoiser.denoise(
    protein=protein,
    verbose=True
) 