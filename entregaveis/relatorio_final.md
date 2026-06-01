# Relatório Final: Avaliação do Modelo TabICL v2

**Disciplina:** Aprendizagem de Máquina
**Modelo Atribuído:** TabICL v2 (Group Model 2)
**Baselines Avaliados:** LightGBM, XGBoost, CatBoost

---

## 1. Resumo Executivo
Este projeto teve como objetivo avaliar o desempenho do foundation model **TabICL v2**, focado em In-Context Learning (ICL) para dados tabulares. O modelo foi testado em 30 datasets extraídos do benchmark TabArena-v0.1 e comparado com métodos estado-da-arte baseados em árvores (LightGBM, XGBoost e CatBoost). 

Os resultados indicam que o TabICL v2 é altamente superior em capacidade preditiva (AUC de 0.928 vs 0.907 do melhor baseline). No entanto, sua arquitetura baseada em atenção possui limitações de escalabilidade para datasets com alta dimensionalidade e grande volume de amostras, resultando em falhas de memória (Out-Of-Memory) em 4 dos 30 datasets testados.

## 2. Metodologia
- **Datasets:** Foram selecionados 30 datasets multiclasse e binários do TabArena-v0.1 via OpenML.
- **Divisão (Split):** Utilizou-se um particionamento estratificado de 70% para treino (contexto) e 30% para teste, garantindo a reprodutibilidade com a semente (seed) fixa em 42.
- **Pré-processamento:** O pipeline foi aprimorado para codificar classes e features categóricas, bem como realizar imputação de valores nulos (NaN) usando a mediana, garantindo que os modelos baseline não falhassem. O TabICL processa essas anomalias de forma nativa.
- **Avaliação:** Foram medidas as métricas AUC OvO (métrica primária), Acurácia, G-Mean, Cross-Entropy e Tempo de Execução.

## 3. Resultados Quantitativos

### 3.1. Médias Agregadas
Nas avaliações agregadas (excluindo os 4 datasets onde ocorreram falhas de hardware), o TabICL v2 obteve a melhor performance preditiva:

| Modelo | AUC OvO Média | Acurácia Média | G-Mean Média |
|---|---|---|---|
| **TabICL v2** | **0.9287** | **0.8787** | **0.8003** |
| CatBoost | 0.9072 | 0.8524 | 0.7312 |
| XGBoost | 0.9016 | 0.8530 | 0.7519 |
| LightGBM | 0.8995 | 0.8525 | 0.6909 |

### 3.2. Análise Estatística (Friedman e Nemenyi)
O teste de Friedman confirmou uma diferença estatisticamente significativa entre os classificadores. O ranking médio (Mean Rank) isola a performance do TabICL como amplamente superior:

1. **TabICL v2:** 1.17
2. **CatBoost:** 2.65
3. **LightGBM:** 2.94
4. **XGBoost:** 3.23

*(Um diagrama de diferença crítica (CD diagram) foi gerado em `results/cd_diagram.png` detalhando estas separações).*

### 3.3. Teste Bayesiano com Região de Equivalência Prática (ROPE)
O teste bayesiano com um ROPE de 0.01 de diferença na métrica AUC comparou o TabICL contra o LightGBM, concluindo:
- **60.0% de probabilidade** de o TabICL ser práticamente superior.
- **40.0% de probabilidade** de ambos serem equivalentes.
- **0.0% de probabilidade** de o TabICL ser inferior.

## 4. Observações e Limitações

Apesar do desempenho excepcional em métricas de classificação, a arquitetura foundation do TabICL v2 revelou um gargalo computacional. 

Em **4 dos 30 datasets** (`isolet`, `jm1`, `adult`, e `Bioresponse`), o modelo falhou durante a execução. O cálculo de inferência (In-Context Learning) aplica uma matriz de atenção entre todas as amostras de treino e teste, resultando em complexidade $O(n^2 \cdot p)$. Nesses 4 datasets (que combinam alta dimensionalidade com milhares de instâncias), o uso de memória escalou para dezenas de gigabytes, causando erros de falta de memória (Out-Of-Memory) no hardware local.

## 5. Conclusão
O TabICL v2 representa um avanço significativo em precisão para problemas tabulares, superando até mesmo os modelos estado-da-arte baseados em Gradient Boosting sem a necessidade de sintonia fina (hyperparameter tuning). 

**Recomendação:** É o modelo ideal para datasets pequenos e médios (< 10.000 instâncias) onde o ganho de AUC compensa o tempo de inferência. No entanto, para grandes volumes de dados, o CatBoost e LightGBM continuam sendo alternativas superiores do ponto de vista de infraestrutura e engenharia de software devido à alta eficiência computacional.
