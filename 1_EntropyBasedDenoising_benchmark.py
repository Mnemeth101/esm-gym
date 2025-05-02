from benchmarking_utils import BenchmarkRunner, get_smallest_pdb_file, save_benchmark_results
from esm.sdk.api import ESM3InferenceClient, ESMProtein
from denoising_strategies import EntropyBasedDenoising
import os

# Initialize the ESM3InferenceClient
from esm.sdk import client
token = os.getenv("ESM_FORGE_API_KEY")
model = client(model="esm3-large-2024-03", url="https://forge.evolutionaryscale.ai", token=token)

# PDB file
pdb_file = "/Users/matthewnemeth/Documents/1_Projects/esm-gym/data/casp15_monomers_without_T1137/T1121-D2.pdb"

# Instantiate the EntropyBasedDenoising strategy with specific parameters
denoising_strategy = EntropyBasedDenoising(
    client=model,
    noise_percentage=50.0,  
    num_decoding_steps=20,    
    temperature=0.5
)

# Create the BenchmarkRunner with just the source protein path
# The runner will load the protein for us
runner = BenchmarkRunner(client=model, source_protein_path=pdb_file, num_runs=50, verbose=True)

# Run the benchmark - just pass the strategy, source protein is already loaded
results = runner.run_benchmark(denoising_strategy, strategy_name="EntropyBasedDenoising")
# results = runner.run_benchmark_parallel(denoising_strategy, strategy_name="EntropyBasedDenoising")

# Save the results
output_file = "data/benchmarking_results/entropy_based_denoising_results.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
save_benchmark_results(runner, output_file=output_file)

print(f"Benchmark results saved to {output_file}")