from benchmarking import BenchmarkRunner
from esm.sdk.api import ESM3InferenceClient, ESMProtein
import os

# Initialize the ESM3InferenceClient
from esm.sdk import client
from denoising_strategies import SimulatedAnnealingDenoising
token = os.getenv("ESM_FORGE_API_KEY")
model = client(model="esm3-large-2024-03", url="https://forge.evolutionaryscale.ai", token=token)

# PDB file - using the same file as in the EntropyBasedDenoising benchmark
pdb_file = "/Users/matthewnemeth/Documents/1_Projects/esm-gym/data/casp15_monomers_without_T1137/T1121-D2.pdb"

# Load the protein using ESMProtein.from_pdb()
source_protein = ESMProtein.from_pdb(pdb_file)

# Instantiate the SimulatedAnnealingDenoising strategy
# Use base_temperature instead of temperature for this strategy
denoising_strategy = SimulatedAnnealingDenoising(
    client=model,
    num_decoding_steps=3,
    noise_percentage=50.0,
    base_temperature=1.0,
)

# Create the BenchmarkRunner with the client and strategy name
runner = BenchmarkRunner(client=model, strategy_name="SimulatedAnnealingDenoising")

if __name__ == '__main__':
    # Run the benchmark - pass the protein and strategy
    # results = runner.run_benchmark(source_protein, denoising_strategy, num_trials=10, verbose=True)
    # Use the parallel version:
    results = runner.run_benchmark_parallel(source_protein, denoising_strategy, num_trials=10, verbose=True)

    # No need to manually save results - they're automatically saved in the job folder
    print(f"Benchmark complete - results saved in data/benchmarking_results/SimulatedAnnealingDenoising_*")