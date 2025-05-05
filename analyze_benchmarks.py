import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv # Add csv import

# Define paths
results_dir = "data/benchmark_results"
figures_dir = os.path.join(results_dir, "figures")

# Create figures directory if it doesn't exist
os.makedirs(figures_dir, exist_ok=True)

# --- Data Loading and Processing ---
all_data = []
aggregated_data = {}

print(f"Loading data from: {results_dir}")

for filename in os.listdir(results_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(results_dir, filename)
        print(f"Processing: {filename}")
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            strategy_name = data.get("strategy", "Unknown Strategy")
            num_trials = data.get("num_trials", 0)
            protein_length = data.get("protein_length", "N/A")

            # Extract trial data
            for trial in data.get("trials", []):
                trial_metrics = trial.get("metrics", {})
                trial_data = {
                    "strategy": strategy_name,
                    "trial": trial.get("trial", None),
                    "runtime": trial.get("runtime", None),
                    **trial_metrics # Unpack metrics like uacce, plddt, ptm, foldability
                }
                all_data.append(trial_data)

            # Extract aggregated data
            agg_metrics = data.get("aggregated_metrics", {})
            aggregated_data[strategy_name] = {
                "diversity": agg_metrics.get("diversity"),
                # Calculate mean for list-based metrics if they exist and are not empty
                "avg_entropy": np.mean(agg_metrics["entropy"]) if agg_metrics.get("entropy") else None,
                "avg_cosine_similarity": np.mean(agg_metrics["cosine_similarity"]) if agg_metrics.get("cosine_similarity") else None,
                "num_trials": num_trials,
                "protein_length": protein_length
            }

        except json.JSONDecodeError:
            print(f"Error decoding JSON from {filename}")
        except Exception as e:
            print(f"An error occurred processing {filename}: {e}")

# Convert trial data to DataFrame
df_trials = pd.DataFrame(all_data)

# Convert aggregated data to DataFrame
df_aggregated = pd.DataFrame.from_dict(aggregated_data, orient='index')

print("\nData loaded successfully.")
print(f"Found data for strategies: {list(df_trials['strategy'].unique())}")
print(f"Total trials processed: {len(df_trials)}")

# --- Analysis and Plotting ---

# Set plot style
sns.set_theme(style="whitegrid")

# 1. Trial Metrics Comparison (Box Plots)
trial_metrics_to_plot = ["uacce", "plddt", "ptm", "foldability", "runtime"]

print("\nGenerating trial metric comparison plots...")
for metric in trial_metrics_to_plot:
    if metric in df_trials.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="strategy", y=metric, data=df_trials, palette="viridis")
        plt.title(f"Distribution of {metric.upper()} per Trial by Strategy")
        plt.xlabel("Denoising Strategy")
        plt.ylabel(metric.upper())
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(figures_dir, f"boxplot_{metric}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")
    else:
        print(f"Skipping plot for '{metric}' - column not found.")


# 2. Aggregated Metrics Comparison (Bar Plots)
aggregated_metrics_to_plot = ["diversity", "avg_entropy", "avg_cosine_similarity"]

print("\nGenerating aggregated metric comparison plots...")
df_aggregated_reset = df_aggregated.reset_index().rename(columns={'index': 'strategy'})

for metric in aggregated_metrics_to_plot:
     if metric in df_aggregated_reset.columns and not df_aggregated_reset[metric].isnull().all():
        plt.figure(figsize=(10, 6))
        sns.barplot(x="strategy", y=metric, data=df_aggregated_reset, palette="viridis", errorbar=None) # Use errorbar=None for direct values
        plt.title(f"Comparison of {metric.replace('_', ' ').title()} by Strategy")
        plt.xlabel("Denoising Strategy")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(figures_dir, f"barplot_{metric}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")
     else:
        print(f"Skipping plot for '{metric}' - column not found or all values are NaN.")

# 3. Average Runtime Comparison (Bar Plot)
print("\nGenerating average runtime comparison plot...")
avg_runtime = df_trials.groupby('strategy')['runtime'].mean().reset_index()

if not avg_runtime.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x="strategy", y="runtime", data=avg_runtime, palette="viridis", errorbar=None)
    plt.title("Average Runtime per Trial by Strategy")
    plt.xlabel("Denoising Strategy")
    plt.ylabel("Average Runtime (seconds)")
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plot_path = os.path.join(figures_dir, "barplot_avg_runtime.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")
else:
    print("Skipping average runtime plot - no runtime data found.")


# --- Summary Statistics ---
print("\n--- Summary Statistics ---")

# Trial Metrics Summary
print("\nTrial Metrics (Mean per Strategy):")
mean_trial_metrics = df_trials.groupby('strategy')[trial_metrics_to_plot].mean()
print(mean_trial_metrics)

# Aggregated Metrics Summary
print("\nAggregated Metrics:")
# Ensure we only select columns that actually exist and contain data
valid_aggregated_metrics = [m for m in aggregated_metrics_to_plot if m in df_aggregated.columns and not df_aggregated[m].isnull().all()]
aggregated_metrics_summary = df_aggregated[valid_aggregated_metrics]
print(aggregated_metrics_summary)

# --- Combine and Save Summary Table ---
print("\nCombining and saving summary table...")
# Combine mean trial metrics and aggregated metrics
summary_table = pd.concat([mean_trial_metrics, aggregated_metrics_summary], axis=1)

# Define the desired order for strategies
desired_order = [
    "OneShotDenoising",
    "EntropyBasedDenoising",
    "MaxProbBasedDenoising",
    "SimulatedAnnealingDenoising"
    # Add any other strategies if needed, or they will be appended/dropped
]

# Reorder the table based on the desired index order
# Keep only strategies present in the original table and in the desired order
available_strategies_in_order = [s for s in desired_order if s in summary_table.index]
# Add any strategies from the table that were not in the desired_order list (optional, uncomment if needed)
# available_strategies_in_order.extend([s for s in summary_table.index if s not in desired_order])

summary_table = summary_table.reindex(available_strategies_in_order)


# Round numeric columns for better readability
numeric_cols = summary_table.select_dtypes(include=np.number).columns
summary_table[numeric_cols] = summary_table[numeric_cols].round(4) # Round to 4 decimal places

# Define the path for the CSV file
csv_path = os.path.join(results_dir, "summary_metrics_table.csv")

try:
    # Save the reordered table
    summary_table.to_csv(csv_path, index_label="strategy", quoting=csv.QUOTE_NONNUMERIC)
    print(f"Summary table saved successfully to: {csv_path}")

    # --- Generate Visual Table Figure (using the reordered table) ---
    print("\nGenerating visual table figure...")
    # Use the already reordered and rounded summary_table DataFrame
    summary_table_loaded = summary_table # No need to reload from CSV

    # Define metrics to maximize and minimize
    metrics_to_maximize = ["uacce", "plddt", "ptm", "foldability", "diversity", "avg_entropy", "avg_cosine_similarity"]
    metrics_to_minimize = ["runtime"]

    # Find best performers (using the reordered table)
    best_performers = {}
    for col in summary_table_loaded.columns:
        # Check if column exists and is numeric before finding max/min
        if col in summary_table_loaded.columns and pd.api.types.is_numeric_dtype(summary_table_loaded[col]):
            if col in metrics_to_maximize:
                # Handle potential NaNs
                if not summary_table_loaded[col].isnull().all():
                    best_performers[col] = summary_table_loaded[col].idxmax()
            elif col in metrics_to_minimize:
                if not summary_table_loaded[col].isnull().all():
                    best_performers[col] = summary_table_loaded[col].idxmin()

    # Create figure
    # Adjust figsize dynamically based on number of columns and rows
    num_rows = len(summary_table_loaded)
    num_cols = len(summary_table_loaded.columns)
    fig_width = max(12, num_cols * 1.5) # Base width 12, add 1.5 per column
    fig_height = max(2.5, num_rows * 0.5) # Base height 2.5, add 0.5 per row
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight')
    ax.axis('off')

    # Create the table using the reordered data
    the_table = ax.table(cellText=summary_table_loaded.values,
                         colLabels=summary_table_loaded.columns,
                         rowLabels=summary_table_loaded.index,
                         cellLoc='center',
                         loc='center')

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    # Adjust scale based on figure size to prevent overlap
    scale_x = 1.2 * (fig_width / 12)
    scale_y = 1.2 * (fig_height / 2.5)
    the_table.scale(scale_x, scale_y)

    # Bold the best performers
    for i, col in summary_table_loaded.columns:
        if col in best_performers:
            best_strategy = best_performers[col]
            # Find row index safely in the reordered index
            if best_strategy in summary_table_loaded.index:
                row_idx = list(summary_table_loaded.index).index(best_strategy)
                cell = the_table[row_idx + 1, i] # +1 because of header row
                cell.set_text_props(weight='bold')

    # Save the figure
    figure_path = os.path.join(figures_dir, "summary_metrics_figure.png")
    plt.savefig(figure_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Visual summary table saved successfully to: {figure_path}")

except Exception as e:
    print(f"Error saving summary table to CSV or generating visual table: {e}")


print("\nAnalysis complete. Figures saved in:", figures_dir)
