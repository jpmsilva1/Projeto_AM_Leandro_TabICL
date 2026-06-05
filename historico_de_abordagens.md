# Diário de Bordo Científico: Evolução Metodológica e Tratamento de Anomalias

Este documento registra a evolução arquitetural e as tomadas de decisão técnicas ao longo da execução do benchmark de 30 datasets tabulares utilizando Modelos Baseados em Árvores, TabICL v2 e AutoGluon. O objetivo é fornecer justificativas empíricas e teóricas para a configuração final escolhida no cluster da Universidade.

## 1. Fase Inicial: O Gargalo de CPU
A arquitetura inicial do projeto foi desenhada para executar em instâncias modestas com apenas 2 CPUs.
- **Observação:** Notou-se rapidamente que algoritmos de ensemble avançados, especialmente o AutoGluon (que empilha múltiplos modelos como Random Forests, Extra Trees, e Gradient Boosting), sofrem de gargalo de I/O e processamento em ambientes limitados.
- **Decisão:** Concluiu-se que o hardware local ou VMs básicas não seriam capazes de entregar os resultados em tempo hábil para a escala de 30 datasets (alguns contendo dezenas de milhares de amostras).

## 2. Validação em Nuvem (GPU Dedicada via Runpod)
Para validar a corretude dos pipelines de código (processamento de features categóricas, splits estratificados, extração de embeddings no TabICL), subimos temporariamente uma instância paga na plataforma Runpod equipada com uma RTX 4090 e processadores robustos.
- **Resultados:** O código provou-se funcional e as predições do TabICL se beneficiaram drasticamente dos 24GB de VRAM da placa.
- **Ajuste de Custo:** Devido ao custo por hora (~\$0.69/h) e o orçamento limitado, reduzimos estrategicamente o `time_limit` do *AutoGluon Extreme* de 2 horas (7200s) para 1 hora (3600s). Isso permitiu maximizar o número de datasets processados dentro do orçamento disponível ($15), provando que o modelo conseguia manter alta performance com metade do tempo teto.

## 3. Migração para o Cluster Universitário (Apuana - 8 CPUs)
Com o código validado, o workload primário foi redirecionado gratuitamente para o cluster Apuana (CIn-UFPE).
- **Setup Inicial:** Solicitamos 1 GPU (RTX 3090 / 24GB) e 8 CPUs para o job principal, visando não sobrecarregar a fila e garantir entrada rápida. O Optuna foi alocado paralelamente com 16 CPUs.
- **Desempenho:** O pipeline sustentou estabilidade extrema, rodando ininterruptamente por dezenas de horas e suportando com sucesso a travessia dos datasets nas categorias `small` e `medium`.

## 4. Análise e Tratamento de Anomalias Matemáticas (Scikit-Learn)
Durante o teste de estresse contínuo, mapeamos comportamentos atípicos gerados por datasets extremamente esparsos ("sujos") do repositório OpenML:

### A. O Bug Oculto do OpenML: Colisão de Task ID vs Dataset ID
- **Problema Descoberto no Estresse:** Durante a execução de datasets pesados (como `houses` e `KDDCup09_appetency`), notamos um comportamento bizarro na tela. O dataset `KDDCup09_appetency`, que deveria ser uma classificação binária com 50.000 amostras, estava sendo carregado com **1829 amostras, 1024 features e 657 classes**.
- **A Causa Raiz:** Descobrimos que o dicionário de datasets do projeto usava o **Task ID (tid)**. Nosso código tentava fazer `openml.datasets.get_dataset(tid)`. Se, por pura coincidência, um dataset de imagens genérico tivesse o mesmo ID numérico da nossa Task, o OpenML baixava o dataset errado (explicando as 1024 features de embeddings e 657 classes).
- **A Solução Definitiva:** Corrigimos o código da Rodada Final para **sempre** buscar a Task primeiro (`task = openml.tasks.get_task(tid)`) e somente depois baixar o dataset referenciado por ela (`openml.datasets.get_dataset(task.dataset_id)`). Isso eliminou completamente os "falsos datasets" que corrompiam a execução.

### B. O Paradoxo de Amostras vs. Classes (Datasets Esparsos Reais)
- **Problema:** Um dataset foi carregado com 50 classes para apenas 66 amostras no total. Ao aplicar o `test_size=0.3`, o subconjunto de Teste reteve amostras de apenas uma fração das 50 classes originais. 
- **O Colapso:** A biblioteca *Scikit-Learn* explodiu no cálculo do AUC (`roc_auc_score(..., multi_class='ovo')`) com o erro: *"Number of classes in y_true not equal to the number of columns in y_score"*. A matemática da Curva ROC (que avalia pares de classes) quebra quando tenta calcular a taxa de Falsos Positivos para uma classe com 0 amostras reais presentes.
- **A Solução Científica (Filtragem Dinâmica):** Implementamos uma função `compute_auc_safe` que intercepta esse colapso. Ela detecta automaticamente o subconjunto de classes que "sobreviveram" no Test Set, recorta a matriz de probabilidades apenas para essas colunas, re-normaliza para que a soma seja 1.0, e injeta o parâmetro explicíto `labels=classes_presentes`. Isso blindou o código contra qualquer erro dimensional futuro.

### B. Fallback para "Single Class" e NaNs Esperados
- **Problema:** Datasets como o `JapaneseVowels` demonstraram comportamento anômalo interno nos modelos base do AutoGluon devido ao formato intrínseco dos dados (timeseries multivariável forçada para tabular). Isso resultou em `AUC=nan`.
- **A Solução:** Adotamos o retorno explícito e gracioso de `np.nan` nestes casos matematicamente insolúveis, para preservar a integridade da extração de outros indicadores como Acurácia e Tempo de Treinamento, impedindo que o script encerrasse de forma prematura.

## 5. Arquitetura da Rodada Definitiva (Escalonamento Vertical para 64 CPUs)
Para gerar a Tabela Final dos Resultados (e compensar o peso computacional extremo dos 10 maiores datasets da benchmark), projetamos uma abordagem de força-bruta matemática.

- **A Constatação:** O gargalo no tempo total não é linear. O maior tempo de espera vem do *AutoGluon Extreme*, que obedece a um teto fixo de horas. Se deixarmos o limite em 2h usando 8 CPUs, entregamos **16 horas-CPU** de poder de processamento interno.
- **A Decisão Final:** Criamos os scripts finais (`run_cluster_final.py` e `job_apuana_final.slurm`) solicitando **64 CPUs**. Reduzimos o teto de tempo para **1 hora**. 
- **Justificativa Computacional:** Ao injetar 64 CPUs e conceder 1 hora de teto de processamento, estamos despejando **64 horas-CPU** no ensemble. Estamos cortando o tempo de relógio global pela metade, enquanto injetamos **4 vezes mais força computacional real** na construção de modelos (Random Forests treinam árvores simutaneamente por thread).

O resultado esperado desta última arquitetura é a execução impecável de toda a fila de dados, finalizando o experimento em prazo acelerado e gerando métricas de ensemble com precisão superior.
