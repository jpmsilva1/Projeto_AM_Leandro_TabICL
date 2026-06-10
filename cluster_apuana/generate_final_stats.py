import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats
from baycomp import SignedRankTest

import warnings
warnings.filterwarnings('ignore')

def demsar_analysis_custom(results, order="descending", output_path=None):
    result = autorank(results, alpha=0.05, verbose=False, order=order)
    ranking = pd.Series(result.rankdf["meanrank"], name="mean_rank").sort_values()

    if output_path is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_stats(result, ax=ax)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    return {"ranking": ranking, "autorank_result": result}

def bayesian_pairwise_custom(results, rope=0.01, is_lower_better=False):
    rows = []
    models = list(results.columns)
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            scores_a = results[a].to_numpy()
            scores_b = results[b].to_numpy()
            
            if is_lower_better:
                # If lower is better, we invert so 'test' logic treats larger = better
                scores_a = -scores_a
                scores_b = -scores_b
                
            test = SignedRankTest(x=scores_a, y=scores_b, rope=rope)
            p_left, p_rope, p_right = test.probs()
            rows.append({
                "model_a": a,
                "model_b": b,
                "p_a_worse": float(p_left),
                "p_equivalent": float(p_rope),
                "p_a_better": float(p_right),
            })
    return pd.DataFrame(rows)

def main():
    print("Iniciando geração estatística final...")
    
    out_dir = Path("cluster_apuana/resultados_estatisticos")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = Path("cluster_apuana/RESULTADO_FINAL_APUANA.csv")
    df = pd.read_csv(csv_path)
    
    # === BLOCO 1: DESEMPENHO MÉDIO GLOBAL ===
    print("\n[Bloco 1] Gerando Médias Globais...")
    metrics = {
        'AUC_OVO': ('descending', 0.01),
        'ACC': ('descending', 0.01),
        'G_Mean': ('descending', 0.01),
        'CE': ('ascending', 0.05),
        'total_time_s': ('ascending', 1.0)
    }
    
    avg_df = df.groupby('model')[list(metrics.keys())].mean().round(4)
    
    # Formata tempo para HH:MM:SS
    def format_time(seconds):
        s = int(seconds)
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"
        
    avg_df['Total_Time (HH:MM:SS)'] = avg_df['total_time_s'].apply(format_time)
    avg_df = avg_df.drop(columns=['total_time_s'])
    
    avg_df.to_markdown(out_dir / "01_Desempenho_Medio_Global.md")
    avg_df.to_csv(out_dir / "01_Desempenho_Medio_Global.csv")
    print(" -> Tabela de Desempenho salva.")
    
    # === BLOCO 2: DEMSAR E BAYESIAN PARA CADA MÉTRICA ===
    print("\n[Bloco 2] Gerando CD Diagrams e Bayesianos...")
    for metric, (order, rope) in metrics.items():
        print(f"  Processando {metric}...")
        pivot = df.pivot(index="dataset", columns="model", values=metric).dropna()
        
        # Demsar CD Diagram
        demsar_analysis_custom(
            pivot, 
            order=order, 
            output_path=out_dir / f"cd_diagram_{metric}.png"
        )
        
        # Bayesian
        is_lower = (order == 'ascending')
        bayes_res = bayesian_pairwise_custom(pivot, rope=rope, is_lower_better=is_lower)
        bayes_res.to_csv(out_dir / f"bayesian_rope_{metric}.csv", index=False)
        
    # === BLOCO 3: METADATA E ESTRATIFICAÇÃO POR REGIME ===
    print("\n[Bloco 3] Gerando tabelas de Regimes e Metadados...")
    
    # Tabela 30 Datasets
    try:
        import openml
        from data.load_tabarena import RECOMMENDED_TASK_IDS
        
        # Para cada dataset em RECOMMENDED_TASK_IDS que a gente rodou, puxar info
        # Mas para ser seguro, ler direto os IDs do run_cluster_final
        import sys
        import multiprocessing
        sys.path.insert(0, str(Path("cluster_apuana").absolute()))
        from run_cluster_final import DATASETS as CLUSTER_DATASETS
        
        meta_rows = []
        print("  Baixando metadados do OpenML para estratificação cruzada...")
        for ds in CLUSTER_DATASETS:
            tid = ds['tid']
            try:
                task = openml.tasks.get_task(tid, download_data=False)
                d = openml.datasets.get_dataset(task.dataset_id, download_data=False, download_qualities=True, download_features_meta_data=True)
                
                n = int(d.qualities.get('NumberOfInstances', 0))
                p = int(d.qualities.get('NumberOfFeatures', 0))
                c = int(d.qualities.get('NumberOfClasses', 0))
                miss = int(d.qualities.get('NumberOfInstancesWithMissingValues', 0))
                
                meta_rows.append({
                    'dataset': ds['name'],
                    'tid': tid,
                    'regime': ds['regime'],
                    'n_samples': n,
                    'n_features': p,
                    'n_classes': c,
                    'has_missing': 'Yes' if miss > 0 else 'No',
                    'type': 'Binary' if c == 2 else 'Multiclass'
                })
            except Exception as e:
                print(f"    Erro OpenML: {ds['name']} -> {e}")
                
        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_markdown(out_dir / "02_Lista_30_Datasets.md", index=False)
        meta_df.to_csv(out_dir / "02_Lista_30_Datasets.csv", index=False)
        
        # Cruzamento (Estratificação)
        df_merged = df.merge(meta_df[['dataset', 'type', 'has_missing']], on='dataset', how='left')
        
        # Tabelas dinâmicas por regime
        for category in ['regime', 'type', 'has_missing']:
            cross_tbl = df_merged.groupby([category, 'model'])['AUC_OVO'].mean().unstack().round(4)
            cross_tbl.to_markdown(out_dir / f"03_Estratificacao_{category}.md")
            cross_tbl.to_csv(out_dir / f"03_Estratificacao_{category}.csv")
            
        print(" -> Tabelas de estratificação geradas com sucesso!")
        
    except Exception as e:
        print(f"Falha ao gerar metadados via OpenML: {e}")
        
    print("\n✅ TUDO PRONTO! Resultados salvos em: cluster_apuana/resultados_estatisticos/")

if __name__ == '__main__':
    main()
