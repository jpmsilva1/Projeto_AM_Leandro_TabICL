# Parecer Técnico de Reprodutibilidade do Código (Arquitetura Final)

Análise minuciosa da arquitetura de código definitiva do projeto (`run_cluster_final.py` e suas dependências diretas), focando em implementações feitas do zero que poderiam usar pacotes Python consolidados e riscos de não-determinismo na execução do cluster Apuana.

---

## 1. Implementações "From Scratch" vs Bibliotecas Prontas

### 🔴 `g_mean_score()` — Implementação Manual

A média geométrica dos recalls por classe foi escrita manualmente no script final usando `np.exp(np.mean(np.log(...)))` com clipping a `1e-12`.

```python
# Implementação manual em run_cluster_final.py:
recalls = recall_score(y_true, y_pred, average=None)
recalls = np.clip(recalls, 1e-12, None)
return float(np.exp(np.mean(np.log(recalls))))
```

**Substituição:** `imblearn.metrics.geometric_mean_score` da biblioteca `imbalanced-learn` faz exatamente isso, com testes unitários e edge-cases cobertos.

---

### 🔴 `preprocess()` — Pipeline Customizado

O pipeline de pré-processamento manual do `run_cluster_final.py` faz encoding categórico, tratamento de NaN e remoção de colunas constantes de forma iterativa usando Pandas.

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

### 🟡 Cross-Validation manual (Optuna)

O `run_cluster_final.py` possui um loop manual de `StratifiedKFold` com `LabelEncoder` embutido para guiar a otimização de hiperparâmetros.

**Substituição:** `sklearn.model_selection.cross_val_score` faz o mesmo com menos código estrutural e menores riscos de bugs de partição.

---

### 🟢 `TimeoutError` customizado

Uma classe `TimeoutError` foi definida manualmente no topo do script do cluster. Desde o Python 3.3, `TimeoutError` já é uma exceção nativa built-in.

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

Na função de construção do AutoGluon (`run_autogluon`), os parâmetros e `presets` são injetados, mas o `random_state` / `seed` não é forçado no framework. A reprodutibilidade do AutoGluon sem uma seed explícita travada no nível do Kernel pode variar levemente dependendo do Bagging interno de Árvores de Decisão.

---

### 🔴 Sem seeding de PyTorch/CUDA para o TabICL

O TabICL roda em GPU via PyTorch dentro do script final, mas em nenhum lugar do fluxo as flags de determinismo absoluto são chamadas:
```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
Isso significa que **execuções consecutivas do benchmark final podem gerar predições levemente diferentes** na placa de vídeo devido a otimizações assíncronas do cuDNN.

---

## 3. Problemas Arquiteturais Secundários

### 🟡 Nenhum framework de logging

A execução monolítica de dezenas de horas no cluster Apuana usa exclusivamente `print()` com emojis para rastreio. A ausência do uso do pacote nativo `logging` impede a persistência estruturada por níveis de severidade e dificulta o debug se a interface SLURM derrubar o processo em background.

### 🟡 Hardcoding e Centralização

Vários parâmetros cruciais da execução final não estão isolados em um arquivo central de configuração, mas "chumbados" no código:
- A semente global (`SEED = 42`)
- Fator de divisão do split (`test_size=0.3`)
- Quantidade de tentativas do Optuna (`N_TRIALS = 50`)
- Tempos de corte de segurança (`time_limit=3600` e `alarm_limit=5400`)

### 🟡 Bloqueio Cego de Exceções (`bare except`)

Blocos genéricos `try... except Exception as e:` são utilizados (ex: linha 425) engolindo praticamente qualquer erro e mascarando stacktraces completos em caso de falhas críticas de sistema.

---

## 4. Tabela de Substituições Recomendadas

| Implementação em `run_cluster_final.py` | Biblioteca Python Recomendada |
|---|---|
| `g_mean_score()` | `imblearn.metrics.geometric_mean_score` |
| Estruturação em `preprocess()` | `sklearn.compose.ColumnTransformer` + `Pipeline` |
| `compute_auc()` com renormalização | `sklearn.metrics.roc_auc_score(labels=...)` |
| Loop de validação do Optuna | `sklearn.model_selection.cross_val_score` |
| Controle de Timeouts de thread | Modulo nativo `concurrent.futures` com parâmetro `timeout=` |
