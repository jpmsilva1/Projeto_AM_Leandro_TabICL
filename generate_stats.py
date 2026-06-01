import pandas as pd
from pathlib import Path
from src.pipeline.stats import demsar_analysis, bayesian_pairwise

def main():
    # Load raw results
    df = pd.read_csv("results/raw.csv")
    
    # Calculate average metrics per model
    print("--- Average Metrics ---")
    
    # Handle optional 'time' column gracefully based on how it was saved
    cols_to_avg = ["auc_ovo", "accuracy"]
    if "g_mean" in df.columns: cols_to_avg.append("g_mean")
    if "cross_entropy" in df.columns: cols_to_avg.append("cross_entropy")
    if "time" in df.columns: cols_to_avg.append("time")
    
    avg = df.groupby("model")[cols_to_avg].mean()
    print(avg)
    
    # Find datasets where group_model failed
    failed_datasets = df[(df['model'] == 'group_model') & (df['auc_ovo'].isna())]['dataset'].tolist()
    if failed_datasets:
        print(f"\n[OBSERVATION] The group_model (TabICL) failed to run on the following {len(failed_datasets)} datasets:")
        for ds in failed_datasets:
            print(f"  - {ds}")
    else:
        print("\n[OBSERVATION] The group_model successfully ran on all datasets!")

    # Prepare data for statistical tests (pivot: dataset x model)
    # We use 'auc_ovo' as the primary metric
    pivot_auc = df.pivot(index="dataset", columns="model", values="auc_ovo")
    
    # Drop rows with NaNs (like isolet where group_model failed) to allow autorank to run
    pivot_auc = pivot_auc.dropna()
    print(f"\nUsing {len(pivot_auc)} datasets for statistical tests (dropped datasets with missing model results).")
    
    # 1. Friedman + Nemenyi CD Diagram
    print("\n--- Running Friedman + Nemenyi ---")
    demsar_res = demsar_analysis(pivot_auc, output_dir=Path("results"))
    print("\nRanking:")
    print(demsar_res["ranking"])
    
    # 2. Bayesian Pairwise Test (ROPE)
    print("\n--- Running Bayesian Pairwise Test ---")
    bayes_res = bayesian_pairwise(pivot_auc)
    bayes_res.to_csv("results/bayesian_rope.csv", index=False)
    print("\nBayesian Results saved to results/bayesian_rope.csv")
    print(bayes_res)

if __name__ == "__main__":
    main()
