from benchmarking import BenchmarkRunner, load_protein_from_fasta
from esm.sdk.api import ESM3InferenceClient, ESMProtein
import os

# Initialize the ESM3InferenceClient
from esm.sdk import client
from denoising_strategies import SimulatedAnnealingDenoising
token = os.getenv("ESM_FORGE_API_KEY")
model = client(model="esm3-large-2024-03", url="https://forge.evolutionaryscale.ai", token=token)

# PDB file - using the same file as in the EntropyBasedDenoising benchmark
pdb_file = "/Users/matthewnemeth/Documents/1_Projects/esm-gym/data/casp15_monomers_without_T1137/T1121-D2.pdb"

# Instantiate the SimulatedAnnealingDenoising strategy
# Use base_temperature instead of temperature for this strategy
denoising_strategy = SimulatedAnnealingDenoising(
    client=model,
    num_decoding_steps=20,
    noise_percentage=50.0,
    base_temperature=1.0,
)

# Create the BenchmarkRunner with just the source protein path
# The runner will load the protein for us
runner = BenchmarkRunner(client=model, source_protein_path=pdb_file, num_runs=10, verbose=True)

if __name__ == '__main__':
    # Run the benchmark - just pass the strategy, source protein is already loaded
    # results = runner.run_benchmark(denoising_strategy, strategy_name="SimulatedAnnealingDenoising")
    # Uncomment to use the parallel version:
    results = runner.run_benchmark_parallel(denoising_strategy, strategy_name="SimulatedAnnealingDenoising")

    # No need to manually save results - they're automatically saved in the job folder
    print(f"Benchmark complete - results saved in data/benchmarking_results/SimulatedAnnealingDenoising_*")