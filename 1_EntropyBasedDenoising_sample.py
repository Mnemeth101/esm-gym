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

# Import the BaseDenoisingStrategy and MaxProbBasedDenoising classes
from denoising_strategies import BaseDenoisingStrategy, MaxProbBasedDenoising, EntropyBasedDenoising
from benchmarking import single_metric_UACCE

## On Forge with larger ESM3 models
from esm.sdk import client
token = os.getenv("ESM_FORGE_API_KEY")
model = client(model="esm3-large-2024-03", url="https://forge.evolutionaryscale.ai", token=token)
# --- Configuration ---
TEST_SEQUENCE = "ACDETSLAQGKACDETSLAQGKACDETSLAQGKACDETSLAQGK"
NOISE_PERCENTAGE = 80.0
NUM_DECODING_STEPS = 21
TEMPERATURE = 0.5
TRACK = "sequence"
# --- End Configuration ---

# Create a dummy protein
protein = ESMProtein(sequence=TEST_SEQUENCE)
print(f"Original Protein: {protein.sequence}\n")

# Instantiate Denoiser with local model and parameters
denoiser = EntropyBasedDenoising(
    model, 
    noise_percentage=NOISE_PERCENTAGE,
    num_decoding_steps=NUM_DECODING_STEPS,
    temperature=TEMPERATURE,
)

# Call denoise with just the protein
denoiser.denoise(protein, verbose=True)
protein, cost = denoiser.return_generation()
print(f"Final Protein: {protein.sequence}\n")
print(f"Cost: {cost}\n")

# Get metrics
print("Calculating metrics...\n")
# Calculate uCCE
ucc = single_metric_UACCE(model, protein)
print(f"uCCE: {ucc}\n")