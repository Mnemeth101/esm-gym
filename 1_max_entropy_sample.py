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
# Import the Tee and PrintFormatter classes from denoising_strategies.py
from denoising_strategies import Tee, PrintFormatter

# Import the BaseDenoising and MaxProbBasedDenoising classes
from denoising_strategies import BaseDenoising, MaxProbBasedDenoising, EntropyBasedDenoising
from benchmarking_utils import UACCE

## On Forge with larger ESM3 models
from esm.sdk import client
token = os.getenv("ESM_FORGE_API_KEY")
model = client(model="esm3-open", url="https://forge.evolutionaryscale.ai", token=token)
# --- Configuration ---
TEST_SEQUENCE = "ACDE"
NOISE_PERCENTAGE = 50.0 # Mask 50% initially (2 positions for length 4)
NUM_DECODING_STEPS = 2 # Number of steps to unmask
TEMPERATURE = 0.0
TRACK = "sequence"
# --- End Configuration ---

# Create a dummy protein
protein = ESMProtein(sequence=TEST_SEQUENCE)
print(f"Original Protein: {protein.sequence}\n")
# Instantiate Denoiser with local model
denoiser = EntropyBasedDenoising(model)
denoiser.track = TRACK # Set track for prints
denoiser.denoise(protein, NOISE_PERCENTAGE, NUM_DECODING_STEPS, TEMPERATURE, TRACK)
protein, cost = denoiser.return_generation()
print(f"Final Protein: {protein.sequence}\n")
print(f"Cost: {cost}\n")

# Get metrics
print("Calculating metrics...\n")
# Calculate uCCE
ucc = UACCE(model, protein)
print(f"uCCE: {ucc}\n")