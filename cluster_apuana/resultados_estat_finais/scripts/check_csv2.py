import pandas as pd

df = pd.read_csv('cluster_apuana/final_run_results_v2.csv')
all_datasets = set(df['dataset'].unique())

print("Missing datasets por modelo:")
for model in df['model'].unique():
    model_datasets = set(df[df['model'] == model]['dataset'].unique())
    missing = all_datasets - model_datasets
    if missing:
        print(f"{model}: {missing}")
