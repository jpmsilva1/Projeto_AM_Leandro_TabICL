# Prévia Consolidada dos Resultados

Este documento compila as tabelas de desempenho global e as estratificações cruzadas (regime, missings e tipo) geradas pelo pipeline estatístico.

## 01 Desempenho Medio Global

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

## 02 Lista 30 Datasets

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

## 03 Estratificacao has missing

| has_missing   |   AutoGluon_Default |   AutoGluon_Extreme |   CatBoost_TD |   CatBoost_Tuned |   LightGBM_TD |   LightGBM_Tuned |   TabICL |   XGBoost_TD |   XGBoost_Tuned |
|:--------------|--------------------:|--------------------:|--------------:|-----------------:|--------------:|-----------------:|---------:|-------------:|----------------:|
| No            |              0.9267 |              0.9313 |        0.9185 |           0.9219 |        0.911  |           0.9244 |   0.9291 |       0.9178 |          0.9225 |
| Yes           |              0.8829 |              0.8769 |        0.8736 |           0.862  |        0.8697 |           0.8478 |   0.873  |       0.8391 |          0.8543 |

---

## 03 Estratificacao regime

| regime   |   AutoGluon_Default |   AutoGluon_Extreme |   CatBoost_TD |   CatBoost_Tuned |   LightGBM_TD |   LightGBM_Tuned |   TabICL |   XGBoost_TD |   XGBoost_Tuned |
|:---------|--------------------:|--------------------:|--------------:|-----------------:|--------------:|-----------------:|---------:|-------------:|----------------:|
| large    |              0.8879 |              0.8674 |        0.8589 |           0.8532 |        0.8422 |           0.8628 |   0.8632 |       0.8497 |          0.859  |
| medium   |              0.9325 |              0.945  |        0.9345 |           0.9354 |        0.9332 |           0.9245 |   0.9427 |       0.9231 |          0.9283 |
| small    |              0.8436 |              0.862  |        0.8309 |           0.8579 |        0.842  |           0.8623 |   0.8702 |       0.8339 |          0.8581 |

---

## 03 Estratificacao type

| type       |   AutoGluon_Default |   AutoGluon_Extreme |   CatBoost_TD |   CatBoost_Tuned |   LightGBM_TD |   LightGBM_Tuned |   TabICL |   XGBoost_TD |   XGBoost_Tuned |
|:-----------|--------------------:|--------------------:|--------------:|-----------------:|--------------:|-----------------:|---------:|-------------:|----------------:|
| Binary     |              0.8797 |              0.8695 |        0.8536 |           0.8563 |        0.8529 |           0.8599 |   0.8621 |       0.8446 |          0.8592 |
| Multiclass |              0.947  |              0.96   |        0.9551 |           0.9511 |        0.9431 |           0.9437 |   0.9618 |       0.9441 |          0.9452 |

---

