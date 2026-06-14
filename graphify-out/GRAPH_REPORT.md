# Graph Report - .  (2026-06-11)

## Corpus Check
- 0 files · ~0 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 263 nodes · 342 edges · 32 communities (20 shown, 12 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 29|Community 29]]

## God Nodes (most connected - your core abstractions)
1. `TabICLSafe` - 12 edges
2. `fit_predict_evaluate()` - 12 edges
3. `stratified_split()` - 11 edges
4. `TabularDataset` - 9 edges
5. `AutoGluonWrapper` - 8 edges
6. `main()` - 7 edges
7. `main()` - 7 edges
8. `demsar_analysis()` - 7 edges
9. `run_autogluon_default()` - 6 edges
10. `load_task()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `build_tabicl()` --inferred--> `TabICLClassifier`  [EXTRACTED]
  cluster_apuana/run_cluster.py → src/models/tabicl_tuning.py
- `build_tabicl()` --inferred--> `TabICLClassifier`  [EXTRACTED]
  cluster_apuana/run_cluster_final.py → src/models/tabicl_tuning.py
- `run_autogluon()` --inferred--> `TabularPredictor`  [EXTRACTED]
  cluster_apuana/run_cluster_final.py → src/models/automl.py
- `build_tabicl()` --inferred--> `TabICLClassifier`  [EXTRACTED]
  kaggle/kaggle_pipeline.py → src/models/tabicl_tuning.py
- `test_split_is_stratified()` --inferred--> `stratified_split()`  [EXTRACTED]
  tests/test_pipeline.py → src/pipeline/split.py

## Import Cycles
- None detected.

## Communities (32 total, 12 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (24): classify_regime(), iter_datasets(), load_task(), DataFrame, Carregamento padronizado dos 30 datasets do TabArena-v0.1.  Os 30 datasets sao s, Itera sobre todos os datasets configurados., Tabela-resumo (n_samples, n_features, n_classes, regime, missing)., Retorna 'small', 'medium' ou 'large' conforme o numero de amostras. (+16 more)

### Community 1 - "Community 1"
Cohesion: 0.10
Nodes (25): _compute_metrics(), EvaluationResult, fit_predict_evaluate(), g_mean_score(), Calculo das metricas exigidas: AUC OVO, ACC, G-Mean, Cross-Entropy e tempo., G-Mean: media geometrica do recall por classe., Retorna (auc_ovo, accuracy, g_mean, cross_entropy) para um conjunto., Treina e avalia um estimador no treino e no teste, medindo tempos. (+17 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (11): AutoGluonWrapper, build_autogluon_default(), build_autogluon_extreme(), _compute_metrics_safe(), evaluate(), g_mean_score(), load_task(), ============================================================================= Cé (+3 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (19): catboost_search_space(), evaluate(), g_mean_score(), install_packages(), lgbm_search_space(), load_task(), main(), preprocess() (+11 more)

### Community 4 - "Community 4"
Cohesion: 0.13
Nodes (10): build_tabicl(), build_tabicl(), build_tabicl_safe(), TabICL com OOM fallback (GPU→CPU) e feature selection automática., Se n_features > MAX_FEATURES, seleciona top-k por variância., TabICLSafe, build_tabicl(), build_group_model() (+2 more)

### Community 5 - "Community 5"
Cohesion: 0.19
Nodes (15): compute_auc(), compute_cross_entropy(), cross_val_objective(), g_mean_score(), main(), objective_cat(), objective_lgb(), objective_xgb() (+7 more)

### Community 6 - "Community 6"
Cohesion: 0.21
Nodes (15): compute_auc(), compute_cross_entropy(), cross_val_objective(), g_mean_score(), main(), objective_cat(), objective_lgb(), objective_xgb() (+7 more)

### Community 7 - "Community 7"
Cohesion: 0.18
Nodes (10): CatBoost_TD_Classifier, LGBM_TD_Classifier, build_catboost(), build_lightgbm(), build_xgboost(), Baselines obrigatorios: LightGBM, XGBoost e CatBoost.  Por padrao usamos as vari, LightGBM com defaults meta-tunados (TD) do pytabkit., XGBoost com defaults meta-tunados (TD) do pytabkit. (+2 more)

### Community 8 - "Community 8"
Cohesion: 0.25
Nodes (9): main(), bayesian_pairwise(), demsar_analysis(), Analise estatistica: Demsar (autorank) e Bayesiana com ROPE (baycomp).  Friedman, Roda Friedman + post-hoc + CD diagram via autorank.      Args:         results:, Roda Bayesian signed-rank test par a par com regiao de equivalencia ROPE.      A, Any, DataFrame (+1 more)

### Community 10 - "Community 10"
Cohesion: 0.22
Nodes (9): build_tabicl_factory(), Espaço de busca Optuna para TabICL v2.  Define o search_space e a factory para u, Amostra hiperparâmetros do TabICL v2 para um trial Optuna., Cria um TabICLClassifier a partir de um dict de hiperparâmetros., Retorna uma factory compatível com tune() que recebe apenas params., tabicl_factory(), tabicl_search_space(), Any (+1 more)

### Community 11 - "Community 11"
Cohesion: 0.24
Nodes (9): export_markdown(), pivot_for_stats(), Geracao de tabelas-resumo a partir do CSV de resultados brutos., Tabela com media e desvio de cada metrica, agrupada por modelo., Pivota para `dataset x modelo` no formato esperado por autorank/baycomp., Exporta a tabela-resumo em Markdown para inclusao no relatorio., summary_by_model(), DataFrame (+1 more)

### Community 12 - "Community 12"
Cohesion: 0.39
Nodes (8): AutoGluon, README Entregáveis, Etapa 1: Fundamentação teórica, Etapa 2: Estudo experimental, TabArena, Relatório Template, Rubrica de Avaliação, Slides Template

### Community 13 - "Community 13"
Cohesion: 0.25
Nodes (7): Busca de hiperparametros com Optuna em validacao cruzada (CV) no treino.  Implem, Roda Optuna no espaco de busca passado e retorna (melhor_params, melhor_score)., tune(), Any, DataFrame, ndarray, Trial

### Community 14 - "Community 14"
Cohesion: 0.67
Nodes (3): ESTADO_DO_PROJETO, relatorio_final, README

## Knowledge Gaps
- **29 isolated node(s):** `DataFrame`, `LGBM_TD_Classifier`, `XGB_TD_Classifier`, `CatBoost_TD_Classifier`, `Trial` (+24 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TabICLClassifier` connect `Community 4` to `Community 10`?**
  _High betweenness centrality (0.372) - this node is a cross-community bridge._
- **Why does `build_group_model()` connect `Community 4` to `Community 0`?**
  _High betweenness centrality (0.241) - this node is a cross-community bridge._
- **Why does `TabICLSafe` connect `Community 4` to `Community 2`?**
  _High betweenness centrality (0.131) - this node is a cross-community bridge._
- **What connects `DataFrame`, `Carregamento padronizado dos 30 datasets do TabArena-v0.1.  Os 30 datasets sao s`, `Retorna 'small', 'medium' ou 'large' conforme o numero de amostras.` to the rest of the system?**
  _78 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.08333333333333333 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.10098522167487685 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.09230769230769231 - nodes in this community are weakly interconnected._