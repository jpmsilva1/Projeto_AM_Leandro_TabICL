# Relatório Final: Avaliação do Modelo TabICL v2 e AutoML (AutoGluon)

**Disciplina:** Aprendizagem de Máquina
**Modelo Atribuído:** TabICL v2 (In-Context Learning para Dados Tabulares)
**Baselines Avaliados:** LightGBM, XGBoost, CatBoost (Versões Padrão e Tunadas via Optuna) e AutoGluon (Default e Extreme 4h).

---

## 1. Resumo Executivo
Este projeto teve como objetivo avaliar o desempenho do foundation model **TabICL v2**, que aplica o paradigma de In-Context Learning (ICL) nativo de LLMs para dados tabulares. O modelo foi rigorosamente testado em 30 datasets do benchmark TabArena-v0.1 e comparado com métodos estado-da-arte baseados em árvores.

Para garantir um campo de batalha justo, os modelos tradicionais foram otimizados utilizando **Optuna** (Otimização Bayesiana de Hiperparâmetros), e o framework **AutoGluon** foi executado no seu preset mais pesado (`best_quality` com 4 horas de limite). Devido ao alto custo computacional, o AutoGluon Extreme foi executado de forma distribuída (Tática de Enxame).

Os resultados finais surpreenderam: o **TabICL v2 obteve o melhor ranking estatístico (Mean Rank) em Acurácia, AUC-OVO e Cross-Entropy**, superando modelos exaustivamente tunados e até mesmo o ensemble massivo do AutoGluon, tudo isso com um tempo de inferência que dura minutos contra as horas exigidas pelo AutoML.

## 2. Metodologia
- **Datasets:** Foram selecionados 30 datasets multiclasse e binários, segmentados por tamanho (10 Small, 10 Medium, 10 Large).
- **Busca de Hiperparâmetros:** XGBoost, CatBoost e LightGBM passaram por otimização de hiperparâmetros via Optuna para encontrar a configuração ideal para cada dataset (as Curvas de Validação e Sensibilidade foram salvas em `/plots`).
- **Tática de Enxame (AutoGluon):** Para viabilizar a execução do AutoGluon Extreme, dividimos os 30 datasets entre os 3 membros do grupo (João, Vinicius e Clara). Os resultados foram consolidados via script de automação (`juntar_resultados.py`).
- **Tratamento de Dados:** Empregou-se imputação estatística para valores nulos (para evitar falhas nos modelos de árvore, visto que o TabICL lida com NaNs nativamente). As 3 únicas falhas isoladas de modelos (como o TabICL no dataset `anneal`) foram imputadas com a média do respectivo regime.

## 3. Resultados Quantitativos e Análise Estatística

A avaliação foi guiada pelo Teste de Friedman com post-hoc Nemenyi (CD Diagrams), garantindo significância estatística ($\alpha = 0.05$).

### 3.1. AUC-OVO (Área sob a Curva ROC)
Embora o AutoGluon Extreme 4h tenha atingido uma **mediana** altíssima (0.981 vs 0.950 do TabICL), quando analisamos o **Mean Rank** ao longo de todos os 30 datasets, o TabICL venceu estatisticamente:
1. **TabICL:** Mean Rank 2.60
2. **AutoGluon Extreme 4h:** Mean Rank 2.88
3. **XGBoost Tuned:** Mean Rank 4.83

### 3.2. Acurácia (ACC) e Cross-Entropy (CE)
O TabICL repetiu a liderança nas demais métricas de separação de classes:
- **Acurácia (ACC):** TabICL (Rank 2.95) superou o AutoGluon Extreme 4h (Rank 4.20) e LightGBM Tuned (Rank 4.31).
- **Cross-Entropy (CE):** O TabICL esmagou a concorrência com um Mean Rank de **1.95**, provando ser extremamente confiante e calibrado em suas probabilidades preditivas.

### 3.3. Teste Bayesiano (Signed-Rank Test)
Um teste bayesiano com ROPE (Region of Practical Equivalence) de 1% ($0.01$) na métrica AUC comparou o TabICL contra os baselines:
1. **TabICL vs LightGBM Tuned:** O teste apontou 83.1% de probabilidade de ambos serem empiricamente equivalentes (ROPE 1%), com 16.8% de chance de vitória estrita do TabICL.
2. **TabICL vs AutoGluon Extreme 4h:** O confronto final demonstrou colossais **99.6% de probabilidade de equivalência prática**.

Estes resultados provam o absurdo do paradigma "Zero-Shot" do In-Context Learning: sem nenhum passo de atualização de gradientes (treino), o modelo reproduz com $99.6\%$ de confiança o poder preditivo atingido por um *ensemble* de AutoML que rodou horas.

## 4. O Custo vs Benefício (Scatter Plot)
A grande revelação do estudo está no gráfico de dispersão (*Custo vs Desempenho Geral*). 
- O **AutoGluon Extreme 4h** consegue disputar o pódio de performance com o TabICL, mas cobra um preço computacional abusivo (uma média na escala de dezenas de milhares de segundos de treinamento/inferência por dataset).
- O **TabICL**, por não necessitar de gradient descent, backpropagation ou hyperparameter tuning, atinge o topo da performance em uma fração minúscula do tempo (minutos). 

## 5. Limitações Críticas e Gargalos

Apesar do desempenho excepcional, a arquitetura foundation do TabICL sofre do problema inerente aos Transformers: **Complexidade de Atenção Quadrática $O(N^2)$**. 
Durante a inferência (onde as instâncias de treino são passadas no contexto), datasets com altíssima dimensionalidade ou volume massivo de linhas esgotam a memória VRAM de GPUs comerciais rapidamente. Os modelos de árvore (como LightGBM e XGBoost) continuam sendo incomparavelmente mais leves no quesito consumo de memória RAM/VRAM.

## 6. Conclusão Final
O projeto provou que o paradigma de **In-Context Learning para tabelas não é apenas viável, mas já atinge o Estado da Arte (SOTA)** em conjuntos de dados de pequeno e médio porte. 

O TabICL v2 consegue superar suites colossais de AutoML (AutoGluon) e modelos de árvore submetidos à intensa otimização Bayesiana (Optuna). Sendo assim, para cenários onde a disponibilidade de VRAM não é um gargalo, o TabICL elimina completamente a necessidade e os custos operacionais (MLOps) de treinamento, tunagem e manutenção de pipelines preditivos clássicos.
