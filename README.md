# Projeto Final de Aprendizagem de Maquina, Repositorio-Template

Estrutura de codigo de referencia para a Etapa 2 do Projeto Final da disciplina de Aprendizagem de Maquina (pos-graduacao). Este template padroniza:

1. Carregamento dos 30 datasets do TabArena-v0.1.
2. Implementacao dos baselines (LightGBM, CatBoost, XGBoost) via `pytabkit`.
3. Integracao com AutoGluon 1.4 nos presets `default` e `extreme` (4 horas).
4. Pipeline de split (70/30 com seed fixa), tuning com Optuna, avaliacao das metricas exigidas e analise por regime.
5. Analise estatistica classica (Friedman, Nemenyi, diagrama de diferenca critica) via `autorank` e Bayesiana com ROPE via `baycomp`.
6. Smoke test que valida cada um dos 10 modelos atribuidos com um dataset pequeno.

## Status do template

A tabela abaixo indica, para cada componente, o que ja esta implementado e validado e o que cabe a cada grupo preencher. As convencoes sao:

- **Pronto** componente implementado e validado por smoke test; nada a fazer.
- **Esqueleto** ha um arquivo funcional, porem generico, que cada grupo deve adaptar ao seu modelo.
- **Placeholder** ha apenas um stub; o grupo precisa implementar.
- **Acao do aluno** entrega que nao esta no template e o grupo deve produzir do zero.

| Componente | Status | O que falta |
|---|---|---|
| `data/load_tabarena.py`, carregamento via OpenML | Pronto | Substituir `RECOMMENDED_TASK_IDS` pela lista oficial de 30 datasets escolhidos (10 pequenos + 10 medios + 10 grandes), conforme `https://tabarena.ai`. |
| `src/models/baselines.py` (LightGBM, XGBoost, CatBoost via pytabkit) | Pronto | Nada. |
| `src/models/automl.py` (AutoGluon 1.4 default e extreme) | Pronto | Instalar `autogluon.tabular` no ambiente (incluido no `pyproject.toml`). |
| `src/models/group_model.py` (modelo principal do grupo) | Placeholder | Implementar `build_group_model(seed)` retornando o estimador atribuido (ver exemplos comentados no proprio arquivo). |
| `src/pipeline/split.py` (70/30 estratificado) | Pronto | Nada. |
| `src/pipeline/tune.py` (Optuna, generico) | Esqueleto | Cada grupo define o `search_space` apropriado para o seu modelo. Para baselines `pytabkit` com defaults TD, o tuning pode ser opcional. |
| `src/pipeline/evaluate.py` (AUC OvO, ACC, G-Mean, CE, tempo) | Pronto | Nada. |
| `src/pipeline/stats.py` (autorank + baycomp com ROPE) | Pronto | Nada. |
| `src/pipeline/regime.py` (quebra por regime) | Pronto | Nada. |
| `src/pipeline/run_all.py` (orquestrador CLI) | Pronto | Cada grupo passa `--include-group-model` apos implementar `group_model.py`. |
| `src/reports/results_table.py` (resumos e exportacao Markdown) | Pronto | Nada. |
| `notebooks/01_eda.ipynb` ate `04_demo_stats_regime.ipynb` | Esqueleto | Executar e adaptar; usar para gerar figuras e tabelas do relatorio. |
| `model_cards/TEMPLATE.md` | Placeholder | Copiar para `model_cards/<seu-modelo>.md` e preencher as 11 secoes (detalhes do modelo, uso pretendido, fatores, metricas com IC 95%, dados de avaliacao, dados de treino e pre-treino, analise quantitativa, consideracoes eticas, avisos e recomendacoes, reprodutibilidade, referencias). |
| `tests/test_pipeline.py` (smoke test) | Pronto e validado (7/7 passando) | Nada. |
| Tabela de selecao dos 30 datasets (no relatorio) | Acao do aluno | Construir tabela com nome, task ID OpenML, n, n_features, n_classes e regime. |
| Relatorio final em PDF | Esqueleto | Estrutura sugerida em `entregaveis/relatorio-template.pdf`. Substituir placeholders e preencher conforme as exigencias da disciplina. |
| Slides da apresentacao (20 minutos: 10 + 10) | Esqueleto | Estrutura sugerida em `entregaveis/slides-template.pdf` (cerca de 21 frames, dois blocos de 10 minutos). |
| Rubrica de avaliacao | Pronto | Disponivel em `entregaveis/rubrica.pdf`. Vincula a nota a entregas concretas, decompondo o esquema 40 + 50 + 10 do PDF da disciplina em criterios com pesos especificos. |

## Modelos atribuiveis aos grupos

| # | Modelo | Toolkit |
|---|---|---|
| 1 | TabPFN-2.5 | `tabpfn` |
| 2 | TabICL v2 | `tabicl` |
| 3 | TabM | `pytabkit` |
| 4 | ModernNCA | `LAMDA-Tabular/TALENT` |
| 5 | RealMLP | `pytabkit` |
| 6 | xRFM | `xrfm` |
| 7 | Mambular | `deeptab` |
| 8 | FT-Transformer | `pytabkit` ou `deeptab` |
| 9 | EBM | `interpret` |
| 10 | Trompt | `pytorch-frame` |

## Quickstart

Pre-requisitos: Python 3.11 ou superior e [`uv`](https://docs.astral.sh/uv/) (recomendado) ou `pip`.

```bash
# clonar o repositorio
git clone <url-do-template>
cd projeto-final-AM-template

# opcao A: uv (recomendado)
uv sync

# opcao B: pip
pip install -e .

# rodar o smoke test (verifica que cada modelo retorna predicao valida)
pytest tests/test_pipeline.py -v
```

Para rodar o experimento completo com o seu modelo atribuido:

```bash
# editar src/models/group_model.py para apontar para o modelo do grupo
# rodar o pipeline em todos os 30 datasets:
python -m src.pipeline.run_all --model group_model --seed 42
```

## Estrutura

```
projeto-final-AM-template/
|- README.md
|- pyproject.toml
|- Dockerfile
|- .python-version
|- data/
|   |- load_tabarena.py
|- src/
|   |- models/
|   |   |- baselines.py
|   |   |- automl.py
|   |   |- group_model.py
|   |- pipeline/
|       |- split.py
|       |- tune.py
|       |- evaluate.py
|       |- stats.py
|       |- regime.py
|       |- run_all.py
|   |- reports/
|       |- results_table.py
|- notebooks/
|   |- 01_eda.ipynb
|   |- 02_demo_baselines.ipynb
|   |- 03_demo_modelo_grupo.ipynb
|   |- 04_demo_stats_regime.ipynb
|- model_cards/
|   |- TEMPLATE.md
|- tests/
    |- test_pipeline.py
```

## Fluxo de trabalho recomendado

1. **EDA inicial:** rodar `notebooks/01_eda.ipynb` para inspecionar os 30 datasets.
2. **Baselines:** rodar `notebooks/02_demo_baselines.ipynb` para confirmar que LightGBM, CatBoost e XGBoost executam end-to-end.
3. **Modelo do grupo:** implementar o wrapper em `src/models/group_model.py` e validar em `notebooks/03_demo_modelo_grupo.ipynb`.
4. **Experimento completo:** rodar `python -m src.pipeline.run_all --seed 42`.
5. **Analise estatistica e por regime:** abrir `notebooks/04_demo_stats_regime.ipynb` e gerar diagrama de diferenca critica e analise Bayesiana.
6. **Model card:** copiar `model_cards/TEMPLATE.md` para `model_cards/<nome-do-modelo>.md` e preencher.

## Reprodutibilidade

1. Seed fixa em todas as etapas (`split`, `tune`, `evaluate`).
2. Versoes fixadas no `pyproject.toml`.
3. Dockerfile opcional disponivel para containerizacao.
4. Saidas intermediarias (resultados por dataset por modelo) sao gravadas em CSV em `results/`.

## Licencas

Todas as bibliotecas utilizadas tem licencas permissivas (MIT, Apache 2.0). A unica excecao e o modelo TabPFN-2.5, cujos pesos sao distribuidos sob licenca nao-comercial; o uso academico em sala esta explicitamente autorizado.

## Suporte

Em caso de duvida tecnica, abrir issue no repositorio do template ou contatar o professor da disciplina.
