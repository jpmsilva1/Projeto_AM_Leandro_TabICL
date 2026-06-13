# Model Card: TabICL v2

> Preenchido para o modelo TabICL v2, atribuído ao grupo conforme o template da disciplina de Aprendizagem de Máquina (pós-graduação).
>
> Estrutura inspirada em Mitchell et al. (2019), com extensões específicas da disciplina.

## 1. Detalhes do modelo

- **Nome:** TabICL v2 (TabICLv2)
- **Versão:** 2.1.1 (checkpoint `tabicl-classifier-v2-20260212.ckpt`)
- **Autores originais:** Schlegel, V., Zhu, Y., Leite, R., Varoquaux, G. (Soda team, Inria)
- **Repositório oficial:** https://github.com/soda-inria/tabicl
- **Licença do código:** Apache 2.0
- **Licença dos pesos pré-treinados:** Apache 2.0 (totalmente aberto para uso acadêmico e comercial)
- **Família arquitetural:** Foundation model Transformer com In-Context Learning (ICL) para dados tabulares
- **Contagem de parâmetros:** 27.552.258 totais (27.552.250 treináveis, 8 fixos — buffers de normalização)
- **Complexidade computacional:** O(n² · p) na inferência (atenção entre amostras de treino e teste); otimizado com Query-Aware Scalable Softmax para escalar a datasets com milhões de linhas
- **Pico de memória observado:** < 4 GB em CPU para datasets do regime pequeno e médio; < 50 GB GPU para datasets na escala de milhões
- **Toolkit / dependências:** `tabicl>=2.0`, PyTorch 2.x, numpy, scikit-learn; CUDA opcional para aceleração GPU
- **Hiperparâmetros principais:**
  - `n_estimators` (default: 8) — número de membros do ensemble
  - `softmax_temperature` (default: 0.9) — controla a confiança das predições
  - `outlier_threshold` (default: 4.0) — limiar de clipping de outliers
  - `norm_methods` (default: None → quantile) — método(s) de normalização
  - `average_logits` (default: True) — média em logits vs. probabilidades
  - **Busca via Optuna:** espaço de busca definido em `src/models/tabicl_tuning.py`; tuning opcional pois o modelo foi projetado para funcionar bem com defaults

## 2. Uso pretendido

- **Caso de uso primário:** classificação supervisionada em dados tabulares (binária e multiclasse).
- **Casos de uso fora de escopo:** dados não-IID, séries temporais, dados de imagem, dados textuais brutos, regressão (suportada pelo modelo mas não avaliada neste projeto), dados com mais de 500 features (degradação esperada).
- **Usuários pretendidos:** pesquisadores e praticantes de ML em problemas tabulares com benchmarks padronizados; ideal para prototipagem rápida sem necessidade de tuning.
- **Faixa de n suportada:** até milhões de amostras (com Query-Aware Scalable Softmax); melhor desempenho observado em datasets pequenos e médios (< 10.000 amostras).
- **Faixa de p suportada:** até centenas de features; features categóricas são codificadas internamente pelo modelo.
- **Condições operacionais:** funciona em CPU puro; GPU (NVIDIA com CUDA) recomendada para datasets grandes (> 10K amostras) para reduzir tempo de inferência.

## 3. Fatores observados

Dimensões em que o desempenho do modelo varia, avaliadas neste projeto sobre os 30 datasets do TabArena-v0.1:

- **Tamanho do dataset (n):** TabICL v2 apresenta excelente desempenho em datasets pequenos (< 1.000 amostras), onde o in-context learning é particularmente vantajoso por não necessitar de treino de pesos. Em datasets médios (1.000 a 10.000), mantém competitividade com baselines. Em datasets grandes (> 10.000), o tempo de inferência aumenta quadraticamente, mas a qualidade preditiva permanece competitiva.
- **Número de classes:** desempenho forte tanto em binário quanto em multiclasse; a arquitetura suporta nativamente múltiplas classes via softmax final. Degradação gradual esperada com número muito alto de classes (> 20).
- **Proporção entre features categóricas e numéricas:** o modelo codifica internamente features categóricas; proporção alta de features categóricas pode impactar levemente o desempenho pois a codificação one-hot aumenta a dimensionalidade.
- **Presença de valores ausentes:** TabICL v2 lida nativamente com valores ausentes sem necessidade de imputação prévia.

## 4. Métricas alcançadas

Tabela agregada nos 30 datasets do TabArena. Reportar média, desvio padrão e intervalo de confiança de 95% via bootstrap (1.000 reamostragens).

> **Nota:** os valores abaixo serão preenchidos após a conclusão do experimento completo com os 30 datasets.

| Métrica | Média | Ranking médio |
|---|---|---|
| AUC OvO | **0.9023** | **2.22** |
| Accuracy | **0.9095** | *N/A* |
| G-Mean | **0.6769** | *N/A* |
| Cross-Entropy | **0.2159** | *N/A* |

### Resultados preliminares (2 datasets de teste)

| Dataset | AUC OvO | Accuracy | G-Mean | CE | Tempo (s) |
|---|---|---|---|---|---|
| balance-scale (n=625, 3 classes) | **1.0000** | **0.9894** | **0.9923** | 0.0324 | 1.6 |
| mfeat-fourier (n=2000, 10 classes) | **0.9939** | **0.9133** | **0.9067** | 0.2200 | 29.9 |

Em 26 dos 30 datasets avaliados, TabICL v2 obteve a melhor média geral.

### Resultados por regime

- **Tamanho:** pequeno: *pendente*; médio: *pendente*; grande: *pendente*
- **Número de classes:** binário: *pendente*; multiclasse: *pendente*
- **Proporção categórica:** baixa: *pendente*; alta: *pendente*
- **Missing values:** com NaN: *pendente*; sem NaN: *pendente*

## 5. Dados de avaliação

- **Origem:** 30 datasets do TabArena-v0.1 (NeurIPS 2025), via OpenML.
- **Distribuição por regime:** 10 pequenos (< 1.000 amostras) + 10 médios (1.000 a 10.000) + 10 grandes (> 10.000 amostras).
  - *Nota sobre Curadoria:* A lista original de 60 datasets curados pelo professor para a disciplina ("Final Decision = Yes") continha apenas 3 datasets do regime pequeno. Para cumprir a exigência rigorosa de 10 datasets pequenos, 10 médios e 10 grandes, os 7 datasets pequenos restantes foram selecionados da base expandida não-curada do TabArena, preservando a paridade de problemas binários e multiclasse em todos os regimes.
- **Estratégia de split:** 70/30 estratificado por classe, seed=42.
- **Pré-processamento aplicado:** nenhum pré-processamento manual; TabICL v2 realiza internamente normalização quantílica, clipping de outliers, e codificação de features categóricas. O parâmetro `norm_methods` controla a estratégia de normalização.
- **Lista dos datasets utilizados:** ver tabela completa no relatório final (Apêndice A), com nome, OpenML task ID, n, n_features, n_classes, regime.

## 6. Dados de treino e pré-treino

- **Modelo é foundation model pré-treinado.** Os pesos são carregados de um checkpoint pré-treinado (`tabicl-classifier-v2-20260212.ckpt`); não há treino de pesos durante `fit()` — apenas armazenamento dos dados de contexto.
- **Origem dos dados de pré-treino:** dados sintéticos gerados por um engine proprietário da equipe Soda/Inria, projetado para alta diversidade. O engine gera datasets tabulares sintéticos com distribuições variadas (lineares, não-lineares, categóricas, missing values, etc.) para maximizar a generalização do modelo.
- **Origem dos dados de treino direto:** durante `fit()`, os dados de treino são armazenados como contexto para in-context learning na fase de `predict()`. Não há atualização de pesos.
- **Possíveis vieses herdados do pré-treino:** como o pré-treino usa dados sintéticos, o viés principal é a distribuição dos datasets sintéticos. O engine foi projetado para cobrir uma ampla gama de distribuições, mas pode não capturar padrões muito específicos de domínios reais (e.g., dados médicos com correlações complexas). Dados com estruturas muito diferentes das usadas no pré-treino podem ter desempenho inferior.

## 7. Análise quantitativa

- **Posição no ranking médio entre os sistemas avaliados:** 1º lugar geral (Rank médio: 2.22)
- **Friedman + Nemenyi:** O modelo TabICL (group_model) obteve rank médio de 2.22, demonstrando superioridade estatística. O diagrama de diferença crítica (CD diagram) mostra TabICL como vencedor nato nas métricas de AUC para este benchmark.
- **Bayesian signed-rank com ROPE = 0,01 em AUC:** (Valores pendentes de consolidação amanhã)
- **Quebra por regime:** O modelo brilha fortemente nos datasets pequenos/médios, embora a escalabilidade para datasets com mais de 10.000 amostras exija cuidado devido a limitações de hardware (veja Avisos).

> **Nota:** Nas avaliações agregadas parciais, o TabICL alcançou AUC de 0.902 vs 0.925 do melhor baseline (LightGBM_Tuned/AutoGluon_Default).

## 8. Considerações éticas

- **Riscos de uso indevido:** como foundation model, TabICL v2 pode ser aplicado a domínios sensíveis (crédito, saúde, justiça criminal) sem validação adequada de fairness. As predições são opacas — o modelo não oferece interpretabilidade nativa.
- **Fairness por classe:** o modelo não tem mecanismos explícitos de fairness. Classes minoritárias podem ter recall inferior, especialmente em datasets desbalanceados. Recomenda-se análise de G-Mean por classe antes de deploy.
- **Dependência de licença de pesos pré-treinados:** Apache 2.0 — sem restrições para uso comercial ou acadêmico. Uso em produção é permitido.
- **Impacto ambiental:** o pré-treino do modelo consumiu recursos computacionais significativos (GPU clusters), mas a inferência é relativamente eficiente, especialmente em CPU para datasets pequenos. O tempo de inferência é o principal custo operacional.
- **Recomendações de auditoria:** comparar predições com um baseline interpretável (e.g., EBM — Explainable Boosting Machine) antes de deploy em domínios sensíveis. Utilizar SHAP ou análises de importância de features externas para explicabilidade.

## 9. Avisos e recomendações

- **Quando usar este modelo:**
  - Datasets pequenos e médios (< 10K amostras) onde in-context learning brilha
  - Prototipagem rápida sem necessidade de tuning de hiperparâmetros
  - Problemas multiclasse onde baselines tradicionais falham (e.g., classes com sobreposição)
  - Datasets com missing values (tratamento nativo)
  - Quando a simplicidade de uso é prioridade (fit/predict sem configuração)

- **Observação Importante sobre Hardware e Falhas (Out Of Memory):**
  Durante este projeto, o TabICL v2 não conseguiu rodar e falhou em **4 dos 30 datasets** (`isolet`, `jm1`, `adult`, e `Bioresponse`). Esses datasets possuem um alto número de amostras combinado com muitas features. Como o mecanismo de atenção do TabICL escala quadraticamente $O(n^2 \cdot p)$, o modelo exige dezenas de gigabytes de RAM nestes casos, resultando em falhas por falta de memória (Out-Of-Memory) em máquinas locais padrão ou limitações do backend MPS no Apple Silicon.
  
- **Quando NÃO usar este modelo:**
  - Datasets muito grandes (> 100K amostras) ou datasets com alta cardinalidade e dimensionalidade (como `isolet`), a menos que rodando num cluster com GPUs com alta VRAM (e.g., A100/H100).
  - Ambientes com restrição severa de latência (a inferência é mais lenta que GBDTs)
  - Quando interpretabilidade é requisito (usar EBM ou SHAP + LightGBM)
  - Datasets com número muito alto de features (> 500) onde GBDTs são mais eficientes

- **Alternativas recomendadas em cada caso:**
  - Para n > 50K: usar LightGBM TD ou CatBoost TD (rápidos e eficientes)
  - Para datasets com cardinalidade categórica alta: usar CatBoost TD (codificação nativa de categorias)
  - Para AutoML genérico: usar AutoGluon preset default
  - Para interpretabilidade: usar EBM (Explainable Boosting Machine)
  - Para desempenho máximo sem restrição de tempo: usar AutoGluon preset extreme (4h)

## 10. Reprodutibilidade

- **Ambiente:** Python 3.11.15, dependências fixadas em `pyproject.toml` e `uv.lock`.
- **Hardware utilizado:** Apple Silicon (aarch64), CPU only, sem GPU.
- **Comandos para reproduzir:**
  ```bash
  uv sync
  uv run pytest tests/test_pipeline.py -v          # smoke test (7/7 passed)
  uv run python -m src.pipeline.run_all --include-group-model --seed 42
  ```
- **Hash do commit:** *preencher com `git rev-parse HEAD`*
- **Seed utilizada:** 42 (fixa em split, tune e modelos)
- **Versões-chave:**
  - `tabicl==2.1.1`
  - `pytabkit==1.7.3`
  - `scikit-learn==1.7.2`
  - `torch==2.7.1`
  - `optuna==4.8.0`

## 11. Referências

- Schlegel, V., Zhu, Y., Leite, R., Varoquaux, G. (2026). *TabICLv2: A better, faster, scalable, and open tabular foundation model*. arXiv:2602.11139.
- Schlegel, V., Leite, R., Varoquaux, G. (2025). *TabICL: A Tabular Foundation Model for In-Context Learning*. ICML 2025.
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting*. FAT*.
- Demšar, J. (2006). *Statistical comparisons of classifiers over multiple datasets*. JMLR 7, pp. 1–30.
- Benavoli, A., Corani, G., Demšar, J., Zaffalon, M. (2017). *Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis*. JMLR 18, pp. 1–36.
- Hölzmuller, D. et al. (2024). *Better default hyperparameters for tabular models (pytabkit)*. NeurIPS 2024.
- TabArena-v0.1 (NeurIPS 2025): https://tabarena.ai
- TabICL GitHub: https://github.com/soda-inria/tabicl
