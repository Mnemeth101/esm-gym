from benchmarking import BenchmarkRunner
from esm.sdk.api import ESM3InferenceClient, ESMProtein
from denoising_strategies import OneShotDenoising # Changed import
import os

# Initialize the ESM3InferenceClient
from esm.sdk import client
token = os.getenv("ESM_FORGE_API_KEY")
model = client(model="esm3-large-2024-03", url="https://forge.evolutionaryscale.ai", token=token)

# PDB file
pdb_file = "/Users/matthewnemeth/Documents/1_Projects/esm-gym/data/casp15_monomers_without_T1137/T1121-D2.pdb"

# Load the protein using ESMProtein.from_pdb()
source_protein = ESMProtein.from_pdb(pdb_file)

# Instantiate the OneShotDenoising strategy with specific parameters
denoising_strategy = OneShotDenoising( # Changed class instantiation
    client=model,
    noise_percentage=50.0,
    # num_decoding_steps is not needed
    temperature=0.5
)

# Create the BenchmarkRunner with the client and strategy name
runner = BenchmarkRunner(client=model, strategy_name="OneShotDenoising") # Changed strategy name

if __name__ == '__main__':
    # Run the benchmark - pass the protein, strategy, and num_trials
    results = runner.run_benchmark(source_protein, denoising_strategy, num_trials=50, verbose=True)
    # Uncomment to use the parallel version:
    # results = runner.run_benchmark_parallel(source_protein, denoising_strategy, num_trials=50, verbose=True, n_processes=8)

    # No need to manually save results - they're automatically saved in the job folder
    print(f"Benchmark complete.") # Changed print statement
