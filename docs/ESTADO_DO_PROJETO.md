# Estado do Projeto — TabICL v2 (Entrega: 10/06/2026)

Este documento descreve o que já está implementado, o que foi alterado recentemente e o que ainda falta fazer antes da entrega.

---

## 1. Visão Geral da Estrutura

```
projeto-final-AM-template/
├── kaggle_pipeline.py       ← Pipeline principal para rodar no Kaggle (GPU)
├── kaggle_continue.py       ← Célula de continuação para retomar execuções no Kaggle
├── generate_stats.py        ← Gera análise estatística a partir do CSV de resultados
├── data/
│   └── load_tabarena.py     ← Carrega datasets do OpenML com cache local
├── src/
│   ├── models/
│   │   ├── baselines.py     ← LightGBM, XGBoost, CatBoost via pytabkit (TD defaults)
│   │   ├── group_model.py   ← TabICL v2 (modelo do grupo)
│   │   ├── tabicl_tuning.py ← Espaço de busca Optuna para o TabICL
│   │   └── automl.py        ← AutoGluon (default e extreme) — ainda não integrado
│   └── pipeline/
│       ├── split.py         ← Split 70/30 estratificado, seed fixa
│       ├── tune.py          ← Motor Optuna genérico (CV no treino)
│       ├── evaluate.py      ← Métricas: AUC OvO, ACC, G-Mean, CE, tempo (treino + teste)
│       ├── stats.py         ← Friedman + Nemenyi (autorank) + Bayesian ROPE (baycomp)
│       ├── regime.py        ← Quebra de resultados por regime (tamanho, classes, etc.)
│       └── run_all.py       ← Orquestrador CLI local (sem GPU)
├── src/reports/
│   └── results_table.py     ← Gera tabelas-resumo e exporta Markdown
├── model_cards/
│   ├── TEMPLATE.md          ← Template do professor (11 seções)
│   └── TabICL_v2.md         ← Model card preenchido (resultados parciais)
├── entregaveis/
│   └── relatorio_final.md   ← Relatório parcialmente preenchido
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_demo_baselines.ipynb
│   ├── 03_demo_modelo_grupo.ipynb
│   └── 04_demo_stats_regime.ipynb
└── tests/
    └── test_pipeline.py     ← Smoke test (baselines + split + métricas)
```

---

## 2. O Que Já Está Implementado

### 2.1 Carregamento dos Datasets
**Arquivo:** `data/load_tabarena.py`

- Carrega qualquer dataset do OpenML por `task_id` com cache local em `./cache/tabarena/`.
- Retorna um `TabularDataset` com: `X`, `y`, `n_samples`, `n_features`, `n_classes`, `n_categorical`, `has_missing`, `regime`.
- A lista `RECOMMENDED_TASK_IDS` contém os 30 datasets escolhidos (10 pequenos + 10 médios + 10 grandes), já validados para o projeto.
- A função `classify_regime(n)` classifica `'small'` (< 1.000), `'medium'` (1.000–10.000) ou `'large'` (> 10.000).

### 2.2 Baselines
**Arquivo:** `src/models/baselines.py`

- **LightGBM**, **XGBoost** e **CatBoost** via `pytabkit`, usando as variantes `_TD_Classifier` (defaults meta-tunados por meta-aprendizado em centenas de datasets — NeurIPS 2024).
- API 100% sklearn-compatível: `.fit(X, y)`, `.predict(X)`, `.predict_proba(X)`.
- Não requerem tuning manual; os defaults são fortemente competitivos.

### 2.3 Modelo do Grupo (TabICL v2)
**Arquivo:** `src/models/group_model.py`

- Usa `TabICLClassifier` do pacote `tabicl>=2.0`.
- Configuração: `device="cuda"`, `batch_size=4`, `offload_mode="disk"`, `kv_cache="repr"`.
- **Importante:** o `offload_mode="disk"` foi adicionado para mitigar erros de memória (OOM) em datasets grandes.

**Arquivo:** `src/models/tabicl_tuning.py`

- Define `tabicl_search_space(trial)` para o Optuna, cobrindo:
  - `n_estimators` (4–32, step=4)
  - `softmax_temperature` (0.3–1.5)
  - `outlier_threshold` (2.0–6.0)
  - `average_logits` (True/False)
  - `norm_methods` (1 ou 2 métodos: quantile, robust, power, none)

### 2.4 Split
**Arquivo:** `src/pipeline/split.py`

- Split estratificado 70% treino / 30% teste com `seed=42`.
- Função `stratified_split(X, y, seed, test_size)`.

### 2.5 Avaliação — com métricas de treino E teste
**Arquivo:** `src/pipeline/evaluate.py`

**O que foi alterado recentemente:** a função `fit_predict_evaluate` agora calcula métricas nos dois conjuntos. O `EvaluationResult` retorna:

| Campo | Descrição |
|---|---|
| `auc_ovo` | AUC One-vs-One **no conjunto de teste** |
| `accuracy` | Acurácia **no teste** |
| `g_mean` | G-Mean **no teste** |
| `cross_entropy` | Cross-Entropy **no teste** |
| `fit_time_s` | Tempo de `.fit()` |
| `predict_time_s` | Tempo de `.predict()` no teste |
| `total_time_s` | fit + predict |
| `train_auc_ovo` | AUC OvO **no conjunto de treino** |
| `train_accuracy` | Acurácia **no treino** |
| `train_g_mean` | G-Mean **no treino** |
| `train_cross_entropy` | Cross-Entropy **no treino** |

Isso atende ao **item 4 dos Procedimentos** do professor: *"Apresentar os resultados finais de treino e teste usando a melhor configuração obtida"*.

### 2.6 Análise Estatística
**Arquivo:** `src/pipeline/stats.py`

- `demsar_analysis(pivot_df, output_dir)`: roda Friedman + Nemenyi + CD diagram via `autorank`.
- `bayesian_pairwise(pivot_df, rope=0.01)`: roda signed-rank Bayesiano par-a-par via `baycomp`.
- `generate_stats.py` na raiz: script standalone que lê o CSV de resultados e gera as análises.

### 2.7 Análise por Regime
**Arquivo:** `src/pipeline/regime.py`

- `assign_regimes(metadata)`: adiciona colunas `regime_size`, `regime_classes`, `regime_cat_share`, `regime_missing` a uma tabela de metadados.
- `aggregate_by_regime(results, metadata, regime_col, metric_col)`: agrega métricas por regime e modelo.

### 2.8 Pipeline do Kaggle
**Arquivos:** `kaggle_pipeline.py` e `kaggle_continue.py`

O `kaggle_continue.py` é o script principal atualmente sendo utilizado para execução na GPU do Kaggle. Inclui:

- **Instalação automática** de dependências.
- **OOM fix** (`TabICLSafe`): tenta GPU → CPU → levanta erro claro.
- **Resume automático**: lê o CSV existente e pula datasets já completos.
- **Tuning via holdout interno** (80/20 dentro do treino) em vez de CV completo, por limitação de tempo na GPU.
- **Salvamento incremental** após cada dataset.
- **Métricas de treino E teste** (alterado recentemente).

**O que foi alterado recentemente nos dois arquivos Kaggle:**
- Função `evaluate()` refatorada: agora computa métricas em treino e teste.
- Helpers `_compute_metrics_safe()` / `_score_set()` extraídos para reutilização.
- Prints `[TRAIN]` e `[TEST]` lado a lado no log.
- NaN rows do `except` atualizados com as colunas `train_*`.

### 2.9 Model Card
**Arquivo:** `model_cards/TabICL_v2.md`

Preenchido com as 11 seções exigidas pelo professor. Seções completas:
- Detalhes do modelo, Uso pretendido, Fatores observados, Considerações éticas, Avisos, Reprodutibilidade, Referências.

Seções com placeholder (dependem dos resultados finais):
- **Seção 4 (Métricas):** tabela com valores finais, desvio padrão, IC 95%.
- **Seção 7 (Análise quantitativa):** ranking final, CD diagram, resultados por regime.

### 2.10 Relatório
**Arquivo:** `entregaveis/relatorio_final.md`

Estrutura criada. Seções 1–4 preenchidas com resultados preliminares. Falta completar com os 30 datasets.

### 2.11 Smoke Test
**Arquivo:** `tests/test_pipeline.py`

- 7 testes cobrindo: baselines rodam sem erro, split é estratificado, seed é reproduzível, métricas presentes no retorno, sanidade do G-Mean.
- **Atenção:** com a adição das colunas `train_*`, o teste `test_evaluation_metrics_present` ficou com o conjunto esperado incompleto. Não vai quebrar (o teste usa `issubset`), mas deve ser atualizado para incluir as novas colunas.

---

## 3. O Que Ainda Falta Fazer

### CRÍTICO (bloqueia a entrega)

#### 3.1 Corrigir erros nos datasets que falham para o TabICL
O TabICL falha em 4 datasets por OOM: `isolet`, `jm1`, `adult`, `Bioresponse`. As opções são:

**Opção A — Trocar os datasets (mais simples):**
Substituir os 4 datasets problemáticos por outros do TabArena-v0.1 do mesmo regime. Atualizar `RECOMMENDED_TASK_IDS` em `data/load_tabarena.py` e em `kaggle_continue.py`.

**Opção B — Forçar CPU com sample cap (mais arriscado):**
No `build_tabicl_safe`, adicionar um limite de amostras para datasets grandes; modelos com n > 10.000 rodam em CPU com um subsample de treino. Isso introduz um viés nos resultados.

**Recomendação:** Opção A, trocar os datasets. É mais limpo e mantém a comparação justa.

#### 3.2 AutoGluon não está integrado nos experimentos
O professor exige comparação com **AutoGluon 1.4** em dois presets: `default` e `extreme` (4h). O arquivo `src/models/automl.py` existe mas **nunca foi integrado** no `kaggle_pipeline.py` nem no `kaggle_continue.py`. Os resultados atuais têm apenas 4 modelos (TabICL + 3 baselines); o projeto exige 6 (+ AutoGluon default + AutoGluon extreme).

O que fazer:
1. Adicionar AutoGluon ao `MODEL_REGISTRY` do `kaggle_continue.py`.
2. O AutoGluon tem API diferente (recebe DataFrame com coluna target, não X/y separados). Será necessário um wrapper simples.
3. O preset `extreme` (4h) é impraticável no Kaggle (timeout). Avaliar se é possível rodar localmente ou reduzir o time limit para 1h.

#### 3.3 Compilar resultados finais dos 30 datasets
O CSV `raw_tuned.csv` precisa ter resultados para todos os 30 datasets e todos os modelos (sem NaN). Depois disso:
1. Rodar `generate_stats.py` para gerar o CD diagram final e o CSV bayesiano.
2. Verificar que `pivot_auc.dropna()` retorna 30 linhas (nenhum dataset com resultado faltando).

---

### IMPORTANTE (qualidade da entrega)

#### 3.4 Análise por regime não está no pipeline do Kaggle
O `src/pipeline/regime.py` existe mas não está sendo chamado em lugar nenhum no fluxo principal. O professor exige análise por:
- Regime de tamanho (small / medium / large)
- Número de classes (binário / multiclasse)
- Proporção de features categóricas (baixa / alta)
- Presença de valores ausentes (sim / não)

O que fazer: criar um script `generate_regime_analysis.py` (ou seção no `generate_stats.py`) que:
1. Carrega o `raw_tuned.csv`.
2. Cruza com os metadados dos datasets (n, n_classes, n_categorical, has_missing).
3. Chama `assign_regimes()` e `aggregate_by_regime()` para cada regime.
4. Salva tabelas e gráficos por regime em `results/regime/`.

#### 3.5 Resultados de treino não estão sendo usados em nenhuma análise
As colunas `train_*` agora existem no CSV, mas `generate_stats.py`, `results_table.py` e o relatório ainda não as utilizam. O mínimo necessário:
- Adicionar uma tabela de comparação treino vs. teste no relatório para discutir overfitting.
- Incluir essa discussão na Seção 7 do model card.

#### 3.6 Atualizar o smoke test
**Arquivo:** `tests/test_pipeline.py`, função `test_evaluation_metrics_present`.

O conjunto `expected` precisa incluir as 4 novas colunas de treino:
```python
expected = {
    "auc_ovo", "accuracy", "g_mean", "cross_entropy",
    "fit_time_s", "predict_time_s", "total_time_s",
    "train_auc_ovo", "train_accuracy", "train_g_mean", "train_cross_entropy",
}
```

#### 3.7 Completar o model card com resultados finais
**Arquivo:** `model_cards/TabICL_v2.md`

Preencher as seções marcadas como `pendente`:
- **Seção 4:** Substituir a tabela de resultados preliminares pelos valores finais (média ± DP, IC 95% via bootstrap).
- **Seção 7:** Adicionar resultados por regime (4 dimensões) e o CD diagram final.

#### 3.8 Completar o relatório
**Arquivo:** `entregaveis/relatorio_final.md`

- Adicionar a **tabela dos 30 datasets** com: nome, task OpenML, n, n_features, n_classes, regime (exigência explícita do professor).
- Completar a **análise por regime** (itens 5.i a 5.iv dos Procedimentos).
- Adicionar discussão sobre **overfitting** (comparativo treino vs. teste).
- Atualizar os resultados finais nas tabelas 3.1 e 3.2 com todos os 30 datasets e os 6 modelos (incluindo AutoGluon).

---

### DESEJÁVEL (melhor nota, não bloqueia)

#### 3.9 Montar os slides
O professor exige ~21 slides em dois blocos de 10 minutos:
- **Bloco 1 (10 min):** Apresentação teórica do TabICL v2 (motivação, arquitetura, aprendizado, limitações).
- **Bloco 2 (10 min):** Experimentos (datasets, resultados, CD diagram, Bayesiano, análise por regime, conclusões).

#### 3.10 Organizar o repositório para entrega
- Checar se o `pyproject.toml` tem todas as versões fixadas.
- Garantir que `results/` tem os CSVs finais e o CD diagram.
- Rodar `pytest tests/test_pipeline.py -v` uma última vez para confirmar que está tudo passando.
- Atualizar o hash do commit no model card (`git rev-parse HEAD`).

---

## 4. Resumo do Status por Critério de Avaliação

| Critério (peso) | Status | O que falta |
|---|---|---|
| **Apresentação teórica do modelo (40%)** | Bom — model card bem preenchido | Montar slides (Bloco 1) |
| **Qualidade dos experimentos (50%)** | Parcial — faltam AutoGluon + 4 datasets corrigidos + análise por regime | Ver itens 3.1, 3.2, 3.3, 3.4 |
| **Qualidade da apresentação (10%)** | Não iniciado | Montar os slides completos |

---

## 5. Ordem de Execução Recomendada

1. **Corrigir os 4 datasets com erro** (trocar por equivalentes, atualizar `RECOMMENDED_TASK_IDS`).
2. **Integrar AutoGluon** no `kaggle_continue.py` com um wrapper simples.
3. **Rodar o experimento completo** no Kaggle com os 30 datasets e os 6 modelos.
4. **Rodar `generate_stats.py`** para gerar CD diagram e CSV bayesiano finais.
5. **Criar `generate_regime_analysis.py`** e rodar análise por regime.
6. **Atualizar o model card** com os resultados finais (Seções 4 e 7).
7. **Completar o relatório** (tabela dos 30 datasets, análise por regime, overfitting).
8. **Atualizar o smoke test** (`test_evaluation_metrics_present`).
9. **Montar os slides** (2 blocos de 10 min, ~21 frames).
10. **Revisar o repositório** e garantir reprodutibilidade.
