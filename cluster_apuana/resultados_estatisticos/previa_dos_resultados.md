# Prévia Consolidada dos Resultados

Este documento compila as tabelas de desempenho global, as estratificações cruzadas (regime, missings e tipo), os gráficos de diferença crítica (Friedman + Nemenyi) e as análises Bayesianas.

## 1. Tabelas de Desempenho e Estratificação

### 01 Desempenho Medio Global

| model             |   AUC_OVO |    ACC |   G_Mean |     CE |   total_time_s |
|:------------------|----------:|-------:|---------:|-------:|---------------:|
| AutoGluon_Default |    0.9094 | 0.9102 |   0.6577 | 0.232  |      1596.9    |
| AutoGluon_Extreme |    0.9108 | 0.9128 |   0.641  | 0.2198 |      3658.33   |
| CatBoost_TD       |    0.8989 | 0.9078 |   0.6796 | 0.2396 |        23.0447 |
| CatBoost_Tuned    |    0.9002 | 0.9096 |   0.6494 | 0.2449 |       782.262  |
| LightGBM_TD       |    0.8938 | 0.9063 |   0.6561 | 0.3237 |        32.1103 |
| LightGBM_Tuned    |    0.8977 | 0.9095 |   0.6801 | 0.2562 |      2899.26   |
| TabICL            |    0.909  | 0.9165 |   0.6889 | 0.2025 |        54.974  |
| XGBoost_TD        |    0.8897 | 0.9064 |   0.6587 | 0.2688 |         1.1657 |
| XGBoost_Tuned     |    0.8982 | 0.9102 |   0.6553 | 0.2331 |       359.059  |

---

### 02 Lista 30 Datasets

| dataset                |    tid | regime   |   n_samples |   n_features |   n_classes | has_missing   | type       |
|:-----------------------|-------:|:---------|------------:|-------------:|------------:|:--------------|:-----------|
| diabetes               |     37 | small    |         768 |            9 |           2 | No            | Binary     |
| anneal                 |      2 | small    |         898 |           39 |           5 | Yes           | Multiclass |
| credit-g               | 168757 | medium   |        1000 |           21 |           2 | No            | Binary     |
| qsar-biodeg            | 359956 | medium   |        1055 |           42 |           2 | No            | Binary     |
| baseball               |   2077 | medium   |        1340 |           17 |           3 | Yes           | Multiclass |
| yeast                  |   2073 | medium   |        1484 |            9 |          10 | No            | Multiclass |
| splice                 |     45 | medium   |        3190 |           61 |           3 | No            | Multiclass |
| Bioresponse            | 359967 | medium   |        3751 |         1777 |           2 | No            | Binary     |
| hypothyroid            |   3011 | medium   |        3772 |           30 |           4 | Yes           | Multiclass |
| hiva_agnostic          |   3892 | medium   |        4229 |         1618 |           2 | No            | Binary     |
| spambase               |     43 | medium   |        4601 |           58 |           2 | No            | Binary     |
| waveform-5000          |     58 | medium   |        5000 |           41 |           3 | No            | Multiclass |
| churn                  | 359968 | medium   |        5000 |           21 |           2 | No            | Binary     |
| page-blocks            |     30 | medium   |        5473 |           11 |           5 | No            | Multiclass |
| optdigits              |     28 | medium   |        5620 |           65 |          10 | No            | Multiclass |
| satimage               |   2074 | medium   |        6430 |           37 |           6 | No            | Multiclass |
| isolet                 |   3481 | medium   |        7797 |          618 |          26 | No            | Multiclass |
| mushroom               |     24 | medium   |        8124 |           23 |           2 | Yes           | Binary     |
| JapaneseVowels         |   3510 | medium   |        9961 |           15 |           9 | No            | Multiclass |
| pendigits              |     32 | large    |       10992 |           17 |          10 | No            | Multiclass |
| nursery                |     26 | large    |       12960 |            9 |           5 | No            | Multiclass |
| letter                 |      6 | large    |       20000 |           17 |          26 | No            | Multiclass |
| houses                 |   3688 | large    |       20640 |            9 |           2 | No            | Binary     |
| Amazon_employee_access | 359979 | large    |       32769 |           10 |           2 | No            | Binary     |
| KDDCup09_appetency     |   3945 | large    |       50000 |          231 |           2 | Yes           | Binary     |
| APSFailure             | 168868 | large    |       76000 |          171 |           2 | Yes           | Binary     |
| KDD98                  | 361329 | large    |       82318 |          478 |           2 | Yes           | Binary     |
| Diabetes130US          | 211986 | large    |      101766 |           50 |           3 | No            | Multiclass |
| porto-seguro           | 360113 | large    |      595212 |           58 |           2 | Yes           | Binary     |

---

### 03 Estratificacao has missing

| has_missing   |   AutoGluon_Default |   AutoGluon_Extreme |   CatBoost_TD |   CatBoost_Tuned |   LightGBM_TD |   LightGBM_Tuned |   TabICL |   XGBoost_TD |   XGBoost_Tuned |
|:--------------|--------------------:|--------------------:|--------------:|-----------------:|--------------:|-----------------:|---------:|-------------:|----------------:|
| No            |              0.9267 |              0.9313 |        0.9185 |           0.9219 |        0.911  |           0.9244 |   0.9291 |       0.9178 |          0.9225 |
| Yes           |              0.8829 |              0.8769 |        0.8736 |           0.862  |        0.8697 |           0.8478 |   0.873  |       0.8391 |          0.8543 |

---

### 03 Estratificacao regime

| regime   |   AutoGluon_Default |   AutoGluon_Extreme |   CatBoost_TD |   CatBoost_Tuned |   LightGBM_TD |   LightGBM_Tuned |   TabICL |   XGBoost_TD |   XGBoost_Tuned |
|:---------|--------------------:|--------------------:|--------------:|-----------------:|--------------:|-----------------:|---------:|-------------:|----------------:|
| large    |              0.8879 |              0.8674 |        0.8589 |           0.8532 |        0.8422 |           0.8628 |   0.8632 |       0.8497 |          0.859  |
| medium   |              0.9325 |              0.945  |        0.9345 |           0.9354 |        0.9332 |           0.9245 |   0.9427 |       0.9231 |          0.9283 |
| small    |              0.8436 |              0.862  |        0.8309 |           0.8579 |        0.842  |           0.8623 |   0.8702 |       0.8339 |          0.8581 |

---

### 03 Estratificacao type

| type       |   AutoGluon_Default |   AutoGluon_Extreme |   CatBoost_TD |   CatBoost_Tuned |   LightGBM_TD |   LightGBM_Tuned |   TabICL |   XGBoost_TD |   XGBoost_Tuned |
|:-----------|--------------------:|--------------------:|--------------:|-----------------:|--------------:|-----------------:|---------:|-------------:|----------------:|
| Binary     |              0.8797 |              0.8695 |        0.8536 |           0.8563 |        0.8529 |           0.8599 |   0.8621 |       0.8446 |          0.8592 |
| Multiclass |              0.947  |              0.96   |        0.9551 |           0.9511 |        0.9431 |           0.9437 |   0.9618 |       0.9441 |          0.9452 |

---

## 2. Gráficos de Diferença Crítica (CD Diagrams)
*O teste de Friedman avalia a significância global. Se houver diferença, o post-hoc de Nemenyi calcula a Distância Crítica (CD). Modelos conectados por uma barra espessa não possuem diferença estatisticamente significante entre si.*

### CD Diagram: AUC_OVO
![CD Diagram AUC_OVO](/Users/joaopms/Downloads/projeto-final-AM-template-main/cluster_apuana/resultados_estatisticos/cd_diagram_AUC_OVO.png)

### CD Diagram: ACC
![CD Diagram ACC](/Users/joaopms/Downloads/projeto-final-AM-template-main/cluster_apuana/resultados_estatisticos/cd_diagram_ACC.png)

### CD Diagram: G_Mean
![CD Diagram G_Mean](/Users/joaopms/Downloads/projeto-final-AM-template-main/cluster_apuana/resultados_estatisticos/cd_diagram_G_Mean.png)

### CD Diagram: CE
![CD Diagram CE](/Users/joaopms/Downloads/projeto-final-AM-template-main/cluster_apuana/resultados_estatisticos/cd_diagram_CE.png)

### CD Diagram: total_time_s
![CD Diagram total_time_s](/Users/joaopms/Downloads/projeto-final-AM-template-main/cluster_apuana/resultados_estatisticos/cd_diagram_total_time_s.png)


---

## 3. Análise Bayesiana Par a Par (ROPE)
*As tabelas abaixo mostram a probabilidade Bayesiana de o Modelo A ser pior, equivalente (dentro do ROPE), ou melhor que o Modelo B.*

### Bayesiana: AUC_OVO

| model_a           | model_b           |   p_a_worse |   p_equivalent |   p_a_better |
|:------------------|:------------------|------------:|---------------:|-------------:|
| AutoGluon_Default | AutoGluon_Extreme |     0       |        0.98002 |      0.01998 |
| AutoGluon_Default | CatBoost_TD       |     0.04254 |        0.95426 |      0.0032  |
| AutoGluon_Default | CatBoost_Tuned    |     0.02884 |        0.96542 |      0.00574 |
| AutoGluon_Default | LightGBM_TD       |     0.05338 |        0.94356 |      0.00306 |
| AutoGluon_Default | LightGBM_Tuned    |     0.00108 |        0.9988  |      0.00012 |
| AutoGluon_Default | TabICL            |     0       |        0.98794 |      0.01206 |
| AutoGluon_Default | XGBoost_TD        |     0.35696 |        0.643   |      4e-05   |
| AutoGluon_Default | XGBoost_Tuned     |     0.00074 |        0.99922 |      4e-05   |
| AutoGluon_Extreme | CatBoost_TD       |     0.15616 |        0.84384 |      0       |
| AutoGluon_Extreme | CatBoost_Tuned    |     0.43284 |        0.56716 |      0       |
| AutoGluon_Extreme | LightGBM_TD       |     0.43078 |        0.56922 |      0       |
| AutoGluon_Extreme | LightGBM_Tuned    |     0.07294 |        0.92706 |      0       |
| AutoGluon_Extreme | TabICL            |     0.00064 |        0.99936 |      0       |
| AutoGluon_Extreme | XGBoost_TD        |     0.67228 |        0.32772 |      0       |
| AutoGluon_Extreme | XGBoost_Tuned     |     0.09576 |        0.90424 |      0       |
| CatBoost_TD       | CatBoost_Tuned    |     0.01308 |        0.97658 |      0.01034 |
| CatBoost_TD       | LightGBM_TD       |     0.0002  |        0.99928 |      0.00052 |
| CatBoost_TD       | LightGBM_Tuned    |     0.0005  |        0.99318 |      0.00632 |
| CatBoost_TD       | TabICL            |     0       |        0.866   |      0.134   |
| CatBoost_TD       | XGBoost_TD        |     0.01508 |        0.9848  |      0.00012 |
| CatBoost_TD       | XGBoost_Tuned     |     0.00182 |        0.99208 |      0.0061  |
| CatBoost_Tuned    | LightGBM_TD       |     0.03336 |        0.96638 |      0.00026 |
| CatBoost_Tuned    | LightGBM_Tuned    |     0.00134 |        0.99346 |      0.0052  |
| CatBoost_Tuned    | TabICL            |     0       |        0.76588 |      0.23412 |
| CatBoost_Tuned    | XGBoost_TD        |     0.07336 |        0.9258  |      0.00084 |
| CatBoost_Tuned    | XGBoost_Tuned     |     0.00082 |        0.99904 |      0.00014 |
| LightGBM_TD       | LightGBM_Tuned    |     0.00028 |        0.99146 |      0.00826 |
| LightGBM_TD       | TabICL            |     0       |        0.69136 |      0.30864 |
| LightGBM_TD       | XGBoost_TD        |     0.01674 |        0.98194 |      0.00132 |
| LightGBM_TD       | XGBoost_Tuned     |     0.0003  |        0.99278 |      0.00692 |
| LightGBM_Tuned    | TabICL            |     0       |        0.97256 |      0.02744 |
| LightGBM_Tuned    | XGBoost_TD        |     0.05218 |        0.94782 |      0       |
| LightGBM_Tuned    | XGBoost_Tuned     |     4e-05   |        0.99978 |      0.00018 |
| TabICL            | XGBoost_TD        |     0.42882 |        0.57118 |      0       |
| TabICL            | XGBoost_Tuned     |     0.10132 |        0.89868 |      0       |
| XGBoost_TD        | XGBoost_Tuned     |     0       |        0.93838 |      0.06162 |

### Bayesiana: ACC

| model_a           | model_b           |   p_a_worse |   p_equivalent |   p_a_better |
|:------------------|:------------------|------------:|---------------:|-------------:|
| AutoGluon_Default | AutoGluon_Extreme |     0       |        0.9998  |      0.0002  |
| AutoGluon_Default | CatBoost_TD       |     0.00226 |        0.9975  |      0.00024 |
| AutoGluon_Default | CatBoost_Tuned    |     0.0002  |        0.99938 |      0.00042 |
| AutoGluon_Default | LightGBM_TD       |     0.01552 |        0.98386 |      0.00062 |
| AutoGluon_Default | LightGBM_Tuned    |     0       |        0.99998 |      2e-05   |
| AutoGluon_Default | TabICL            |     0       |        0.98896 |      0.01104 |
| AutoGluon_Default | XGBoost_TD        |     0.00594 |        0.99368 |      0.00038 |
| AutoGluon_Default | XGBoost_Tuned     |     2e-05   |        0.99984 |      0.00014 |
| AutoGluon_Extreme | CatBoost_TD       |     0.00658 |        0.99342 |      0       |
| AutoGluon_Extreme | CatBoost_Tuned    |     0.00012 |        0.99988 |      0       |
| AutoGluon_Extreme | LightGBM_TD       |     0.01092 |        0.98908 |      0       |
| AutoGluon_Extreme | LightGBM_Tuned    |     0.00086 |        0.99914 |      0       |
| AutoGluon_Extreme | TabICL            |     0       |        0.9999  |      0.0001  |
| AutoGluon_Extreme | XGBoost_TD        |     0.01842 |        0.98158 |      0       |
| AutoGluon_Extreme | XGBoost_Tuned     |     0.00016 |        0.99984 |      0       |
| CatBoost_TD       | CatBoost_Tuned    |     0       |        0.99954 |      0.00046 |
| CatBoost_TD       | LightGBM_TD       |     8e-05   |        0.99992 |      0       |
| CatBoost_TD       | LightGBM_Tuned    |     0       |        0.99996 |      4e-05   |
| CatBoost_TD       | TabICL            |     0       |        0.83132 |      0.16868 |
| CatBoost_TD       | XGBoost_TD        |     0.00056 |        0.99944 |      0       |
| CatBoost_TD       | XGBoost_Tuned     |     0       |        0.99992 |      8e-05   |
| CatBoost_Tuned    | LightGBM_TD       |     0.001   |        0.99898 |      2e-05   |
| CatBoost_Tuned    | LightGBM_Tuned    |     0       |        1       |      0       |
| CatBoost_Tuned    | TabICL            |     0       |        0.92948 |      0.07052 |
| CatBoost_Tuned    | XGBoost_TD        |     0.00122 |        0.99876 |      2e-05   |
| CatBoost_Tuned    | XGBoost_Tuned     |     0       |        1       |      0       |
| LightGBM_TD       | LightGBM_Tuned    |     0       |        0.99666 |      0.00334 |
| LightGBM_TD       | TabICL            |     0       |        0.89342 |      0.10658 |
| LightGBM_TD       | XGBoost_TD        |     0.00012 |        0.99924 |      0.00064 |
| LightGBM_TD       | XGBoost_Tuned     |     6e-05   |        0.99542 |      0.00452 |
| LightGBM_Tuned    | TabICL            |     0       |        0.94252 |      0.05748 |
| LightGBM_Tuned    | XGBoost_TD        |     0.00086 |        0.99898 |      0.00016 |
| LightGBM_Tuned    | XGBoost_Tuned     |     0       |        1       |      0       |
| TabICL            | XGBoost_TD        |     0.14446 |        0.85554 |      0       |
| TabICL            | XGBoost_Tuned     |     0.02712 |        0.97288 |      0       |
| XGBoost_TD        | XGBoost_Tuned     |     0       |        0.99964 |      0.00036 |

### Bayesiana: G_Mean

| model_a           | model_b           |   p_a_worse |   p_equivalent |   p_a_better |
|:------------------|:------------------|------------:|---------------:|-------------:|
| AutoGluon_Default | AutoGluon_Extreme |     0.00766 |        0.90618 |      0.08616 |
| AutoGluon_Default | CatBoost_TD       |     0.0651  |        0.45656 |      0.47834 |
| AutoGluon_Default | CatBoost_Tuned    |     0.00912 |        0.96792 |      0.02296 |
| AutoGluon_Default | LightGBM_TD       |     0.01216 |        0.768   |      0.21984 |
| AutoGluon_Default | LightGBM_Tuned    |     0.0167  |        0.8987  |      0.0846  |
| AutoGluon_Default | TabICL            |     0.00042 |        0.42702 |      0.57256 |
| AutoGluon_Default | XGBoost_TD        |     0.00692 |        0.75032 |      0.24276 |
| AutoGluon_Default | XGBoost_Tuned     |     0.00948 |        0.75274 |      0.23778 |
| AutoGluon_Extreme | CatBoost_TD       |     0.01992 |        0.91648 |      0.0636  |
| AutoGluon_Extreme | CatBoost_Tuned    |     0.032   |        0.9623  |      0.0057  |
| AutoGluon_Extreme | LightGBM_TD       |     0.00816 |        0.85348 |      0.13836 |
| AutoGluon_Extreme | LightGBM_Tuned    |     0.1638  |        0.79914 |      0.03706 |
| AutoGluon_Extreme | TabICL            |     0.0007  |        0.95038 |      0.04892 |
| AutoGluon_Extreme | XGBoost_TD        |     0.08206 |        0.66448 |      0.25346 |
| AutoGluon_Extreme | XGBoost_Tuned     |     0.02854 |        0.92752 |      0.04394 |
| CatBoost_TD       | CatBoost_Tuned    |     0.18586 |        0.79944 |      0.0147  |
| CatBoost_TD       | LightGBM_TD       |     0.01616 |        0.8964  |      0.08744 |
| CatBoost_TD       | LightGBM_Tuned    |     0.21418 |        0.77798 |      0.00784 |
| CatBoost_TD       | TabICL            |     0.01894 |        0.5291  |      0.45196 |
| CatBoost_TD       | XGBoost_TD        |     0.00922 |        0.87672 |      0.11406 |
| CatBoost_TD       | XGBoost_Tuned     |     0.01716 |        0.94566 |      0.03718 |
| CatBoost_Tuned    | LightGBM_TD       |     0.01464 |        0.80176 |      0.1836  |
| CatBoost_Tuned    | LightGBM_Tuned    |     0.00456 |        0.99374 |      0.0017  |
| CatBoost_Tuned    | TabICL            |     0.00136 |        0.1384  |      0.86024 |
| CatBoost_Tuned    | XGBoost_TD        |     0.00024 |        0.82128 |      0.17848 |
| CatBoost_Tuned    | XGBoost_Tuned     |     0.00014 |        0.99182 |      0.00804 |
| LightGBM_TD       | LightGBM_Tuned    |     0.12356 |        0.86588 |      0.01056 |
| LightGBM_TD       | TabICL            |     0.109   |        0.33972 |      0.55128 |
| LightGBM_TD       | XGBoost_TD        |     0.03032 |        0.94882 |      0.02086 |
| LightGBM_TD       | XGBoost_Tuned     |     0.0375  |        0.94726 |      0.01524 |
| LightGBM_Tuned    | TabICL            |     0.0012  |        0.2639  |      0.7349  |
| LightGBM_Tuned    | XGBoost_TD        |     0.01622 |        0.69854 |      0.28524 |
| LightGBM_Tuned    | XGBoost_Tuned     |     0.0011  |        0.9689  |      0.03    |
| TabICL            | XGBoost_TD        |     0.78146 |        0.1546  |      0.06394 |
| TabICL            | XGBoost_Tuned     |     0.51908 |        0.46782 |      0.0131  |
| XGBoost_TD        | XGBoost_Tuned     |     0.00352 |        0.995   |      0.00148 |

### Bayesiana: CE

| model_a           | model_b           |   p_a_worse |   p_equivalent |   p_a_better |
|:------------------|:------------------|------------:|---------------:|-------------:|
| AutoGluon_Default | AutoGluon_Extreme |     0       |        0.99968 |      0.00032 |
| AutoGluon_Default | CatBoost_TD       |     0.00024 |        0.99868 |      0.00108 |
| AutoGluon_Default | CatBoost_Tuned    |     0.00098 |        0.99828 |      0.00074 |
| AutoGluon_Default | LightGBM_TD       |     0.27252 |        0.72598 |      0.0015  |
| AutoGluon_Default | LightGBM_Tuned    |     0.00218 |        0.99748 |      0.00034 |
| AutoGluon_Default | TabICL            |     0       |        0.99618 |      0.00382 |
| AutoGluon_Default | XGBoost_TD        |     0.10124 |        0.89702 |      0.00174 |
| AutoGluon_Default | XGBoost_Tuned     |     0       |        0.99922 |      0.00078 |
| AutoGluon_Extreme | CatBoost_TD       |     0.00222 |        0.99778 |      0       |
| AutoGluon_Extreme | CatBoost_Tuned    |     0.00106 |        0.99894 |      0       |
| AutoGluon_Extreme | LightGBM_TD       |     0.22396 |        0.77604 |      0       |
| AutoGluon_Extreme | LightGBM_Tuned    |     0.00496 |        0.99504 |      0       |
| AutoGluon_Extreme | TabICL            |     0       |        1       |      0       |
| AutoGluon_Extreme | XGBoost_TD        |     0.14882 |        0.85118 |      0       |
| AutoGluon_Extreme | XGBoost_Tuned     |     8e-05   |        0.99992 |      0       |
| CatBoost_TD       | CatBoost_Tuned    |     0.0014  |        0.9986  |      0       |
| CatBoost_TD       | LightGBM_TD       |     0.0075  |        0.9925  |      0       |
| CatBoost_TD       | LightGBM_Tuned    |     0.00144 |        0.99852 |      4e-05   |
| CatBoost_TD       | TabICL            |     0       |        0.96486 |      0.03514 |
| CatBoost_TD       | XGBoost_TD        |     0.01648 |        0.98352 |      0       |
| CatBoost_TD       | XGBoost_Tuned     |     0       |        0.99996 |      4e-05   |
| CatBoost_Tuned    | LightGBM_TD       |     0.13856 |        0.8561  |      0.00534 |
| CatBoost_Tuned    | LightGBM_Tuned    |     0.00286 |        0.99584 |      0.0013  |
| CatBoost_Tuned    | TabICL            |     0       |        0.98436 |      0.01564 |
| CatBoost_Tuned    | XGBoost_TD        |     0.0627  |        0.93498 |      0.00232 |
| CatBoost_Tuned    | XGBoost_Tuned     |     6e-05   |        0.99886 |      0.00108 |
| LightGBM_TD       | LightGBM_Tuned    |     0.00138 |        0.97588 |      0.02274 |
| LightGBM_TD       | TabICL            |     0       |        0.25798 |      0.74202 |
| LightGBM_TD       | XGBoost_TD        |     0       |        0.9968  |      0.0032  |
| LightGBM_TD       | XGBoost_Tuned     |     0       |        0.96146 |      0.03854 |
| LightGBM_Tuned    | TabICL            |     0       |        0.84214 |      0.15786 |
| LightGBM_Tuned    | XGBoost_TD        |     0.015   |        0.9834  |      0.0016  |
| LightGBM_Tuned    | XGBoost_Tuned     |     0       |        0.99918 |      0.00082 |
| TabICL            | XGBoost_TD        |     0.6859  |        0.3141  |      0       |
| TabICL            | XGBoost_Tuned     |     0.00306 |        0.99694 |      0       |
| XGBoost_TD        | XGBoost_Tuned     |     0       |        0.9525  |      0.0475  |

### Bayesiana: total_time_s

| model_a           | model_b           |   p_a_worse |   p_equivalent |   p_a_better |
|:------------------|:------------------|------------:|---------------:|-------------:|
| AutoGluon_Default | AutoGluon_Extreme |     1       |        0       |      0       |
| AutoGluon_Default | CatBoost_TD       |     0       |        0       |      1       |
| AutoGluon_Default | CatBoost_Tuned    |     0       |        0       |      1       |
| AutoGluon_Default | LightGBM_TD       |     0       |        0       |      1       |
| AutoGluon_Default | LightGBM_Tuned    |     0.2095  |        0       |      0.7905  |
| AutoGluon_Default | TabICL            |     0       |        0       |      1       |
| AutoGluon_Default | XGBoost_TD        |     0       |        0       |      1       |
| AutoGluon_Default | XGBoost_Tuned     |     0       |        0       |      1       |
| AutoGluon_Extreme | CatBoost_TD       |     0       |        0       |      1       |
| AutoGluon_Extreme | CatBoost_Tuned    |     0       |        0       |      1       |
| AutoGluon_Extreme | LightGBM_TD       |     0       |        0       |      1       |
| AutoGluon_Extreme | LightGBM_Tuned    |     0.00726 |        0       |      0.99274 |
| AutoGluon_Extreme | TabICL            |     0       |        0       |      1       |
| AutoGluon_Extreme | XGBoost_TD        |     0       |        0       |      1       |
| AutoGluon_Extreme | XGBoost_Tuned     |     0       |        0       |      1       |
| CatBoost_TD       | CatBoost_Tuned    |     1       |        0       |      0       |
| CatBoost_TD       | LightGBM_TD       |     0.0234  |        0       |      0.9766  |
| CatBoost_TD       | LightGBM_Tuned    |     1       |        0       |      0       |
| CatBoost_TD       | TabICL            |     0.08814 |        0       |      0.91186 |
| CatBoost_TD       | XGBoost_TD        |     0       |        0       |      1       |
| CatBoost_TD       | XGBoost_Tuned     |     1       |        0       |      0       |
| CatBoost_Tuned    | LightGBM_TD       |     0       |        0       |      1       |
| CatBoost_Tuned    | LightGBM_Tuned    |     0.63208 |        0       |      0.36792 |
| CatBoost_Tuned    | TabICL            |     0       |        0       |      1       |
| CatBoost_Tuned    | XGBoost_TD        |     0       |        0       |      1       |
| CatBoost_Tuned    | XGBoost_Tuned     |     0       |        0       |      1       |
| LightGBM_TD       | LightGBM_Tuned    |     1       |        0       |      0       |
| LightGBM_TD       | TabICL            |     0.67912 |        0       |      0.32088 |
| LightGBM_TD       | XGBoost_TD        |     0       |        0.00326 |      0.99674 |
| LightGBM_TD       | XGBoost_Tuned     |     1       |        0       |      0       |
| LightGBM_Tuned    | TabICL            |     0       |        0       |      1       |
| LightGBM_Tuned    | XGBoost_TD        |     0       |        0       |      1       |
| LightGBM_Tuned    | XGBoost_Tuned     |     6e-05   |        0       |      0.99994 |
| TabICL            | XGBoost_TD        |     0       |        0       |      1       |
| TabICL            | XGBoost_Tuned     |     0.99998 |        0       |      2e-05   |
| XGBoost_TD        | XGBoost_Tuned     |     1       |        0       |      0       |

