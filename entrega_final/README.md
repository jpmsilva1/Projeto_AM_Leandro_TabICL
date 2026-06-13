# Entrega Final - Reprodutibilidade

Este diretório contém os artefatos higienizados e essenciais para a reprodutibilidade integral dos experimentos de *Machine Learning* executados ao longo deste projeto, em conformidade com as exigências da rubrica da disciplina.

## 1. Organização do Ambiente Isolado

Para blindar qualquer divergência futura, os três scripts Python finais dependem de um ecossistema estritamente travado. O ambiente foi estabilizado utilizando o gerenciador ultrarrápido **uv**:

- `pyproject.toml`: Declara explicitamente as versões da biblioteca `tabicl`, Scikit-Learn, Optuna e AutoGluon vigentes durante o experimento.
- `uv.lock`: Manifesto de travamento (*lockfile*) profundo. Garante que qualquer build subsequente resolva os mesmos hashes e as exatas subdependências utilizadas na pesquisa.

## 2. Controle de Semente Estocástica (Seed)

O critério de isolamento exige que os testes sejam determinísticos.
Em todos os códigos presentes na pasta `src/`, a constante global foi declarada no escopo principal:
```python
SEED = 42
```
Esta âncora foi transmitida a todos os motores probabilísticos das bibliotecas:
- K-Folds Estratificados do `scikit-learn`
- Permutações de ramificação das árvores (XGBoost, LightGBM, CatBoost)
- Amostrador Bayesiano TPE (*Tree-structured Parzen Estimator*) do `optuna`
- Sementes de hardware e fallback (Torch `deterministic=True` no backend do TabICL).

## 3. Instruções de Reprodução (Modo 1: Local via UV)

Se o seu sistema hospedeiro já possui o utilitário `uv` instalado nativamente, siga os passos abaixo dentro desta pasta:

1. Instrua o gerenciador a instalar rigorosamente as dependências imutáveis do *lockfile*:
   ```bash
   uv sync --frozen
   ```

2. Dispare o pipeline de inferência da arquitetura usando o Python isolado do *.venv* recém criado:
   ```bash
   uv run python src/run_cluster_final_v2.py
   ```

## 4. Instruções de Reprodução (Modo 2: Total via Docker)

Caso necessite de um ambiente "imaculado", que não compartilhe núcleo, rotas Python ou bibliotecas base do seu S.O., utilize o `Dockerfile` incluso. A imagem é baseada no `python:3.11-slim` do Debian, o que erradica conflitos de compilador C++ para as dependências de Machine Learning.

1. Construa o container isolado (O build fará o trigger interno do `uv sync --frozen`):
   ```bash
   docker build -t am_projeto_final .
   ```

2. Execute o container para deflagrar os experimentos em background:
   ```bash
   docker run --rm am_projeto_final
   ```

---

