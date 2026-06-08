# Tabelas Organizadas: Resultados Preliminares (Rodada 1)

> [!NOTE]
> Estes são os resultados parciais retirados do arquivo `cluster_results.csv` gerado antes da correção do OpenML. As linhas com `FAILED` e `-` serão perfeitamente preenchidas pela rodada final que está executando agora no Apuana.

## Tabela de Acurácia (ACC)
| Dataset | AutoGluon | CatBoost_TD | LightGBM_TD | TabICL v2 | XGBoost_TD |
|---|---|---|---|---|---|
| Amazon_employee_access | 0.9502 | 0.9480 | 0.9458 | 0.9492 | 0.9484 |
| Bioresponse | 0.7806 | 0.7806 | 0.7833 | 0.7789 | 0.7771 |
| JapaneseVowels | FAILED | FAILED | FAILED | FAILED | FAILED |
| KDDCup09_appetency | - | - | FAILED | FAILED | FAILED |
| anneal | 0.9778 | 0.9667 | 0.9778 | 0.9704 | 0.9667 |
| baseball | 0.9527 | 0.9428 | 0.9453 | 0.9478 | 0.9502 |
| blood-transfusion-service-center | 0.8000 | 0.7956 | 0.7467 | 0.7956 | 0.7378 |
| churn | 0.9673 | 0.9640 | 0.9633 | 0.9673 | 0.9613 |
| credit-g | 0.7667 | 0.7633 | 0.7667 | 0.7667 | 0.7400 |
| diabetes | 0.7273 | 0.7446 | 0.7359 | 0.7403 | 0.7532 |
| hiva_agnostic | FAILED | FAILED | FAILED | FAILED | FAILED |
| houses | FAILED | FAILED | FAILED | FAILED | FAILED |
| hypothyroid | 0.9947 | 0.9973 | 0.9965 | 0.9947 | 0.9965 |
| isolet | FAILED | FAILED | FAILED | FAILED | FAILED |
| letter | 0.9783 | 0.9655 | 0.9645 | 0.9900 | 0.9585 |
| mushroom | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| nursery | 0.9987 | 1.0000 | 0.8850 | 0.9995 | 0.9997 |
| optdigits | 0.9899 | 0.9858 | 0.9840 | 0.9970 | 0.9763 |
| page-blocks | 0.9756 | 0.9775 | 0.9744 | 0.9781 | 0.9744 |
| pendigits | 0.9921 | 0.9912 | 0.9915 | 0.9976 | 0.9885 |
| qsar-biodeg | 0.8801 | 0.8864 | 0.8864 | 0.8864 | 0.8644 |
| satimage | 0.9295 | 0.9222 | 0.9228 | 0.9419 | 0.9228 |
| spambase | 0.6848 | 0.6739 | 0.6630 | 0.6957 | 0.6304 |
| splice | 0.9561 | 0.9519 | 0.9530 | 0.9655 | 0.9592 |
| waveform-5000 | 0.9697 | 0.9495 | 0.9091 | 0.9933 | 0.8721 |
| yeast | 0.6166 | 0.6054 | 0.5830 | 0.6256 | 0.6143 |

## Tabela de AUC (AUC_OVO)
| Dataset | AutoGluon | CatBoost_TD | LightGBM_TD | TabICL v2 | XGBoost_TD |
|---|---|---|---|---|---|
| Amazon_employee_access | 0.8495 | 0.8110 | 0.8209 | 0.8523 | 0.8278 |
| Bioresponse | 0.8678 | 0.8564 | 0.8614 | 0.8532 | 0.8558 |
| JapaneseVowels | FAILED | FAILED | FAILED | FAILED | FAILED |
| KDDCup09_appetency | - | - | FAILED | FAILED | FAILED |
| anneal | 0.9988 | 0.9985 | 0.9988 | 0.9994 | 0.9979 |
| baseball | 0.9155 | 0.9066 | 0.9088 | 0.9275 | 0.9136 |
| blood-transfusion-service-center | 0.7698 | 0.7382 | 0.7244 | 0.7727 | 0.7129 |
| churn | 0.9351 | 0.9381 | 0.9346 | 0.9444 | 0.9438 |
| credit-g | 0.7992 | 0.7899 | 0.7602 | 0.7885 | 0.7619 |
| diabetes | 0.8379 | 0.8362 | 0.8027 | 0.8384 | 0.7906 |
| hiva_agnostic | FAILED | FAILED | FAILED | FAILED | FAILED |
| houses | FAILED | FAILED | FAILED | FAILED | FAILED |
| hypothyroid | 0.9457 | 0.9602 | 0.9596 | 0.9626 | 0.8205 |
| isolet | FAILED | FAILED | FAILED | FAILED | FAILED |
| letter | 0.9999 | 0.9997 | 0.9996 | 1.0000 | 0.9996 |
| mushroom | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| nursery | 1.0000 | 1.0000 | 0.7834 | 1.0000 | 0.9563 |
| optdigits | 0.9999 | 0.9999 | 0.9998 | 1.0000 | 0.9996 |
| page-blocks | 0.9826 | 0.9796 | 0.9796 | 0.9853 | 0.9803 |
| pendigits | 0.9995 | 0.9995 | 0.9993 | 0.9996 | 0.9995 |
| qsar-biodeg | 0.9277 | 0.9285 | 0.9180 | 0.9339 | 0.9134 |
| satimage | 0.9939 | 0.9928 | 0.9922 | 0.9962 | 0.9916 |
| spambase | 0.6131 | 0.6406 | 0.6419 | 0.6100 | 0.6403 |
| splice | 0.9955 | 0.9951 | 0.9956 | 0.9971 | 0.9960 |
| waveform-5000 | 0.9981 | 0.9980 | 0.9938 | 1.0000 | 0.9894 |
| yeast | 0.8858 | 0.8420 | 0.8306 | 0.8950 | 0.8615 |
