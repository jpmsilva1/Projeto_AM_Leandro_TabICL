import pandas as pd
from autorank import autorank

df = pd.read_csv('cluster_apuana/final_run_results_v2.csv')
df = df[df['model'] != 'AutoGluon_Extreme'].copy()

# imputar furos para não quebrar a média justa (TabICL no anneal)
means = df[(df['model'] == 'TabICL') & (df['regime'] == 'small')].mean(numeric_only=True)
df = pd.concat([df, pd.DataFrame([{
    'model': 'TabICL', 'dataset': 'anneal', 'regime': 'small',
    'ACC': means['ACC'], 'AUC_OVO': means['AUC_OVO'], 'G_Mean': means['G_Mean'],
    'CE': means['CE'], 'total_time_s': means['total_time_s']
}])], ignore_index=True)

# mean metrics for TabICL
t_metrics = df[df['model'] == 'TabICL'][['AUC_OVO', 'ACC', 'G_Mean', 'CE']].mean()
print("TabICL Metrics:")
print(t_metrics)

# autorank
pivot_auc = df.pivot(index='dataset', columns='model', values='AUC_OVO')
# need to drop AutoGluon_Default missing house, XGBoost_Tuned missing nursery just for this quick rank check
pivot_auc.fillna(pivot_auc.mean(), inplace=True)
res = autorank(pivot_auc, alpha=0.05, verbose=False)
print("\nRankings:")
print(res.rankdf)
