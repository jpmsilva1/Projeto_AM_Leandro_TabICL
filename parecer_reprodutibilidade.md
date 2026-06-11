# Parecer Técnico de Reprodutibilidade do Código

Análise minuciosa de todo o código-fonte do projeto, focando em implementações feitas do zero que poderiam usar pacotes Python consolidados, riscos de não-determinismo e problemas arquiteturais.

---

## 1. Implementações "From Scratch" vs Bibliotecas Prontas

### 🔴 `g_mean_score()` — Reimplementada em 3 arquivos

A média geométrica dos recalls por classe foi escrita manualmente usando `np.exp(np.mean(np.log(...)))` com clipping a `1e-12`.

```python
# O que temos (manual, em evaluate.py, run_cluster_final.py e run_optuna.py):
recalls = recall_score(y_true, y_pred, average=None)
recalls = np.clip(recalls, 1e-12, None)
return float(np.exp(np.mean(np.log(recalls))))
```

**Substituição:** `imblearn.metrics.geometric_mean_score` da biblioteca `imbalanced-learn` faz exatamente isso, com testes unitários e edge-cases cobertos.

```python
from imblearn.metrics import geometric_mean_score
g_mean = geometric_mean_score(y_true, y_pred)
```

---

### 🔴 `preprocess()` — Reimplementada em 3 arquivos

Pipeline de pré-processamento manual com encoding categórico, tratamento de NaN e remoção de colunas constantes, copy-pastado entre `run_cluster_final.py`, `run_cluster.py` e `run_optuna.py`.

**Substituição:** `sklearn.compose.ColumnTransformer` + `sklearn.pipeline.Pipeline`:
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ("num", SimpleImputer(strategy="median"), num_cols),
])
```

---

### 🟡 `compute_auc()` — Re-normalização manual de probabilidades

Recorte manual da matriz de probabilidades para classes presentes no test set, seguido de re-normalização linha a linha.

**Substituição:** O parâmetro `labels=` do `sklearn.metrics.roc_auc_score` já trata essa situação de forma nativa e menos propensa a erros.

---

### 🟡 `format_time()` — Conversão manual de segundos para HH:MM:SS

```python
# O que temos (manual):
h = s // 3600; m = (s % 3600) // 60; sec = s % 60

# O que existe pronto:
str(datetime.timedelta(seconds=int(s)))
```

---

### 🟡 Cross-Validation manual em `run_cluster_final.py`

Loop manual de `StratifiedKFold` com `LabelEncoder` por fold.

**Substituição:** `sklearn.model_selection.cross_val_score` faz exatamente o mesmo com menos código e menos riscos de bugs.

---

### 🟢 `TimeoutError` customizado — Sombra do built-in

Uma classe `TimeoutError` foi definida manualmente nos scripts do cluster. Desde o Python 3.3, `TimeoutError` já é um built-in.

---

### 🟢 Closure manual em `tabicl_tuning.py`

`build_tabicl_factory` usa uma closure manual onde `functools.partial` seria mais idiomático.

---

## 2. Problemas Críticos de Reprodutibilidade

### 🔴 Vazamento de Dados (Data Leakage) no Pré-processamento

> [!CAUTION]
> Este é o achado mais grave da auditoria.

Em nosso código consolidado definitivo (`run_cluster_final.py`), o pré-processamento vaza estatísticas do conjunto de teste para a etapa de treinamento.

**Onde exatamente o vazamento ocorre?**
A ordem de execução no loop principal (linhas 391-396) está estruturada da seguinte forma:

```python
# 1. Carrega o dataset inteiro (Treino + Teste juntos)
X, y, cat_indicator, _ = dataset.get_data(...)

# 2. Roda o pré-processamento NO DATASET INTEIRO
X_clean, y_clean, le = preprocess(X, y, cat_indicator)

# 3. SÓ DEPOIS faz a divisão de Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X_clean.values, ...)
```

E dentro da função `preprocess()` (linha 92), executamos:
`X = X.fillna(X.median()).fillna(0)`

**A Mecânica do Leak:**
A biblioteca calcula a **mediana** (`X.median()`) olhando para o conjunto de dados *inteiro* — ou seja, a matemática "espia" os valores que deveriam estar isolados apenas para a validação no futuro. O mesmo vale para o encoding das variáveis categóricas (`cat.codes`), que mapeia categorias presentes apenas no conjunto de teste.

A forma matematicamente "à prova de balas" exigiria que o `train_test_split` ocorresse primeiro, e a mediana fosse extraída exclusivamente de `X_train` para preencher os buracos tanto de `X_train` quanto de `X_test`.

**Impacto Científico:** Os resultados reportados tornam-se otimisticamente enviesados. Embora a diferença final na acurácia seja matematicamente microscópica, esta é uma vulnerabilidade clássica que revisores de bancas acadêmicas procuram apontar. É aconselhável documentar isso no TCC como uma limitação mapeada do pipeline.

---

### 🔴 Seed do AutoGluon nunca é aplicada

Em [automl.py](file:///Users/joaopms/Downloads/projeto-final-AM-template-main/src/models/automl.py), o parâmetro `seed` é aceito na assinatura da função mas **nunca repassado** ao `TabularPredictor`. O AutoGluon recebe `seed` via `.fit()`, mas quem chama `build_autogluon()` recebe a falsa segurança de que passou a seed.

---

### 🔴 Sem seeding de PyTorch/CUDA para o TabICL

O TabICL roda em GPU via PyTorch, mas em nenhum lugar do código são chamados:
```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
Isso significa que **execuções consecutivas do mesmo experimento podem gerar resultados ligeiramente diferentes** na GPU.

---

### 🟡 Teste Bayesiano (`baycomp`) sem seed

Em [stats.py](file:///Users/joaopms/Downloads/projeto-final-AM-template-main/src/pipeline/stats.py), o `SignedRankTest` usa amostragem MCMC internamente, mas **nenhuma seed é passada**. O parâmetro `random_state` existe na API do `baycomp` mas não é utilizado. Isso significa que os valores de `p_a_better` podem variar marginalmente entre execuções.

---

## 3. Problemas Arquiteturais

### 🔴 Duplicação massiva de código

As funções `preprocess()`, `g_mean_score()`, `compute_auc()` e `compute_cross_entropy()` estão **copy-pastadas identicamente** em 3 arquivos:

| Função | run_cluster.py | run_cluster_final.py | run_optuna.py |
|---|---|---|---|
| `preprocess()` | ✅ | ✅ | ✅ |
| `g_mean_score()` | ❌ | ✅ | ✅ |
| `compute_auc()` | ✅ | ✅ | ✅ |
| `compute_cross_entropy()` | ❌ | ✅ | ✅ |
| `cross_val_objective()` | ❌ | ✅ | ✅ |

**Risco:** Se um bug é corrigido num arquivo mas não nos outros, os resultados ficam inconsistentes (e de fato já existem diferenças sutis entre as versões — por exemplo, `run_cluster.py` não limpa NaNs do target).

**Solução ideal:** Extrair para um módulo compartilhado `cluster_apuana/utils.py`.

---

### 🟡 Inconsistência na ordem de carregamento do OpenML

| Arquivo | Ordem |
|---|---|
| `run_cluster_final.py` | `get_task()` primeiro → depois `get_dataset()` |
| `run_optuna.py` | `get_dataset()` primeiro → `get_task()` como fallback |
| `run_cluster.py` | Mistura das duas |

Essa inconsistência pode causar o carregamento de datasets **completamente diferentes** entre os scripts (o famoso bug de colisão TaskID/DatasetID que documentamos no histórico).

---

### 🟡 Tuning em CPU vs Avaliação em GPU

O `run_optuna.py` não configura `device='cuda'` nem `task_type='GPU'` para XGBoost e CatBoost, enquanto o `run_cluster_final.py` (avaliação final) configura. Isso significa que os hiperparâmetros foram otimizados num regime computacional (CPU) e avaliados noutro (GPU), podendo gerar resultados numéricos ligeiramente diferentes.

---

### 🟡 Nenhum framework de logging

Todos os 12+ arquivos Python usam exclusivamente `print()` com emojis. Não há uso do módulo `logging` do Python, o que impede filtragem por nível de severidade, timestamps automáticos e redirecionamento para arquivo.

---

### 🟡 Sem arquivo de configuração centralizado

Parâmetros críticos estão espalhados e hardcoded em vários locais:

| Parâmetro | Onde está | Valor |
|---|---|---|
| Seed | 6 arquivos diferentes | `42` |
| Test size | `split.py`, `run_cluster_final.py` | `0.30` |
| N trials Optuna | `run_cluster_final.py`, `run_optuna.py`, `tune.py` | `50` |
| ROPE | `stats.py`, `generate_final_stats.py` | `0.01` |
| Alpha | `stats.py`, `generate_final_stats.py` | `0.05` |
| Regime thresholds | `load_tabarena.py` | `1000 / 10000` |
| Time limits AutoGluon | `run_cluster_final.py` | `3600 / 1800` |

**Solução ideal:** Um `config.yaml` ou `dataclass` centralizado.

---

## 4. Resumo por Severidade

### Crítico 🔴
| # | Achado | Impacto |
|---|---|---|
| 1 | **Data leakage** no encoding categórico (LabelEncoder fit em train+test) | Métricas otimisticamente enviesadas |
| 2 | **Seed do AutoGluon ignorada** — parâmetro aceito mas nunca repassado | Resultados do AutoGluon não-reprodutíveis |
| 3 | **Sem seeding PyTorch/CUDA** para o TabICL | Resultados GPU não-reprodutíveis |
| 4 | **Duplicação de código** em 3 scripts (com divergências sutis) | Risco de inconsistência nos resultados |
| 5 | **`device="cuda"` hardcoded** sem fallback para CPU | Código não-portável |

### Alto 🟠
| # | Achado | Impacto |
|---|---|---|
| 6 | `g_mean_score()` manual vs `imblearn` | Risco de edge-case não coberto |
| 7 | `preprocess()` manual vs `sklearn.Pipeline` | Código frágil e não-padronizado |
| 8 | Tuning em CPU vs avaliação em GPU | Hiperparâmetros sub-ótimos para o hardware final |

### Médio 🟡
| # | Achado | Impacto |
|---|---|---|
| 9 | Teste Bayesiano sem seed | Resultados estatísticos marginalmente variáveis |
| 10 | Ordem de carregamento OpenML inconsistente entre scripts | Risco de dataset errado |
| 11 | Sem `logging` (só `print()`) | Debugging e auditoria prejudicados |
| 12 | Sem config centralizado | Parâmetros duplicados e divergentes |
| 13 | `bare except:` engolindo exceções | Erros silenciados |
| 14 | `warnings.filterwarnings("ignore")` generalizado | Avisos importantes suprimidos |

### Baixo 🟢
| # | Achado | Impacto |
|---|---|---|
| 15 | `TimeoutError` sombra do built-in | Cosmético |
| 16 | `functools.partial` não usado | Estilo |
| 17 | Imports não utilizados em `run_optuna.py` | Limpeza de código |
| 18 | Tipo de retorno errado em `automl.py` | Documentação |

---

## 5. Tabela de Substituições Recomendadas

| Código Manual | Biblioteca Python Recomendada |
|---|---|
| `g_mean_score()` | `imblearn.metrics.geometric_mean_score` |
| `preprocess()` pipeline | `sklearn.compose.ColumnTransformer` + `Pipeline` |
| `compute_auc()` com renormalização | `sklearn.metrics.roc_auc_score(labels=...)` |
| Cross-validation manual | `sklearn.model_selection.cross_val_score` |
| `format_time()` | `datetime.timedelta(seconds=s)` |
| CSV-based checkpointing | `mlflow`, `wandb` ou `sacred` |
| `TimeoutError` customizado | Built-in `TimeoutError` (Python 3.3+) |
| Thread-limit via `os.environ` | Biblioteca `threadpoolctl` |
| Closure manual (`build_tabicl_factory`) | `functools.partial` |
