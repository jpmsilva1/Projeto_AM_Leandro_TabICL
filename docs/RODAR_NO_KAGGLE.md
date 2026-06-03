# Rodar o Experimento no Kaggle

Use este guia para executar o projeto do jeito que o professor pediu: 30 datasets, 6 modelos, melhores configurações, métricas de treino e teste, análise estatística final.

## 1. Configurar o Notebook

1. Crie um notebook novo no Kaggle.
2. Em `Settings`, deixe:
   - `Accelerator`: GPU T4 x2
   - `Internet`: On
3. Apague as células padrão.

## 2. Célula 1: Setup

Cole e rode esta célula:

```python
import subprocess, sys

packages = [
    "openml>=0.14,<1.0",
    "tabicl>=2.0,<3.0",
    "optuna>=3.6,<5.0",
    "autorank>=1.2,<2.0",
    "baycomp>=1.0,<2.0",
    "catboost>=1.2,<2.0",
    "xgboost>=2.0,<4.0",
    "lightgbm>=4.3,<5.0",
    "autogluon.tabular[all]>=1.4,<2.0",
]

subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages, check=False)
print("Setup concluído")
```

Avisos de conflito com pacotes do RAPIDS/CUDA do Kaggle podem aparecer. Se a última linha for `Setup concluído`, siga para a próxima célula.

## 3. Célula 2: Pipeline

Cole todo o conteúdo de `kaggle_continue.py` na célula 2 e rode.

O script salva tudo em:

```text
/kaggle/working/results/
```

Arquivos esperados:

```text
raw_tuned.csv
best_params.csv
cd_diagram.png
bayesian_rope.csv
```

## 4. Se o Kaggle Parar

Se aparecer `You've hit your session limit`, não é erro do código. É limite temporário de GPU da conta Kaggle.

Quando o horário indicado pelo Kaggle chegar, rode novamente a célula 2. O script lê `raw_tuned.csv` e refaz apenas os datasets incompletos ou com `NaN`.

## 5. Se o OpenML Der 503/504

Também não é erro do código. É instabilidade do servidor OpenML.

O `load_task` tenta novamente automaticamente. Se mesmo assim algum dataset for pulado, rode a célula 2 outra vez mais tarde; o resume cuida do resto.

## 6. Depois de Baixar os Resultados

Coloque os arquivos baixados do Kaggle na pasta `results/` local e rode:

```bash
python generate_stats.py
```

Depois complete:

- `model_cards/TabICL_v2.md`
- `entregaveis/relatorio_final.md`
- slides da apresentação

