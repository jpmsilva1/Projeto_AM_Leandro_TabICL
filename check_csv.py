import pandas as pd

df = pd.read_csv('cluster_apuana/final_run_results_v2.csv')
print("Modelos presentes:")
print(df['model'].unique())

print("\nDatasets por modelo:")
for model in df['model'].unique():
    count = df[df['model'] == model]['dataset'].nunique()
    print(f"{model}: {count} datasets")
