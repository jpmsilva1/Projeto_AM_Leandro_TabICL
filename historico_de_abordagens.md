# Diário de Bordo Científico: Evolução Metodológica e Tratamento de Anomalias

Este documento registra a evolução arquitetural e as tomadas de decisão técnicas ao longo da execução do benchmark de 30 datasets tabulares utilizando Modelos Baseados em Árvores, TabICL v2 e AutoGluon. O objetivo é fornecer justificativas empíricas e teóricas para a configuração final escolhida no cluster da Universidade.

## 1. Fase Inicial: O Gargalo de CPU
A arquitetura inicial do projeto foi desenhada para executar em instâncias modestas com apenas 2 CPUs.
- **Observação:** Notou-se rapidamente que algoritmos de ensemble avançados, especialmente o AutoGluon (que empilha múltiplos modelos como Random Forests, Extra Trees, e Gradient Boosting), sofrem de gargalo de I/O e processamento em ambientes limitados.
- **Decisão:** Concluiu-se que o hardware local ou VMs básicas não seriam capazes de entregar os resultados em tempo hábil para a escala de 30 datasets (alguns contendo dezenas de milhares de amostras).

## 2. Validação em Nuvem (GPU Dedicada via Runpod)
Para validar a corretude dos pipelines de código (processamento de features categóricas, splits estratificados, extração de embeddings no TabICL), subimos temporariamente uma instância paga na plataforma Runpod equipada com uma RTX 4090 e processadores robustos.
- **Resultados:** O código provou-se funcional e as predições do TabICL se beneficiaram drasticamente dos 24GB de VRAM da placa.
- **Orçamento e Ajuste de Custo:** Inserimos $20 dólares de crédito na plataforma com a meta de otimizar a execução para custar apenas $15. Para isso, reduzimos estrategicamente o `time_limit` do *AutoGluon Extreme* de 2 horas (7200s) para 1 hora (3600s). Contudo, devido à altíssima exigência computacional de certos datasets, todo o crédito de $20 foi consumido e, mesmo assim, a execução foi interrompida antes de processar todos os datasets da fila.

## 3. Migração para o Cluster Universitário (Apuana - 8 CPUs)
Com o código validado, o workload primário foi redirecionado gratuitamente para o cluster Apuana (CIn-UFPE).
- **Setup Inicial:** Solicitamos 1 GPU (RTX 3090 / 24GB) e 8 CPUs para o job principal, visando não sobrecarregar a fila e garantir entrada rápida. O Optuna foi alocado paralelamente com 16 CPUs.
- **Desempenho:** O pipeline sustentou estabilidade extrema, rodando ininterruptamente por dezenas de horas e suportando com sucesso a travessia dos datasets nas categorias `small` e `medium`.

## 4. Análise e Tratamento de Anomalias Matemáticas (Scikit-Learn)
Durante o teste de estresse contínuo, mapeamos comportamentos atípicos gerados por datasets extremamente esparsos ("sujos") do repositório OpenML:

### A. Inconsistência na API do OpenML: Sobreposição de Task ID e Dataset ID
- **Anomalia Identificada:** Durante o processamento de bases extensas (como `houses` e `KDDCup09_appetency`), observou-se uma divergência estrutural. O dataset `KDDCup09_appetency`, esperado como uma classificação binária de 50.000 amostras, foi instanciado com **1829 amostras, 1024 features e 657 classes**.
- **Causa Raiz:** A análise apontou que o dicionário interno utilizava o identificador da tarefa (**Task ID - tid**) como argumento direto para a função `openml.datasets.get_dataset(tid)`. Devido à arquitetura do OpenML, IDs numéricos de tarefas e datasets podem colidir, resultando no download de conjuntos de dados não correlacionados (como tarefas de visão computacional).
- **A Solução Definitiva:** Corrigimos o código da Rodada Final para **sempre** buscar a Task primeiro (`task = openml.tasks.get_task(tid)`) e somente depois baixar o dataset referenciado por ela (`openml.datasets.get_dataset(task.dataset_id)`). Isso eliminou completamente os "falsos datasets" que corrompiam a execução.

### B. O Paradoxo de Amostras vs. Classes (Datasets Esparsos Reais)
- **Problema:** Um dataset foi carregado com 50 classes para apenas 66 amostras no total. Ao aplicar o `test_size=0.3`, o subconjunto de Teste reteve amostras de apenas uma fração das 50 classes originais. 
- **Falha Computacional:** A biblioteca *Scikit-Learn* apresentou erro de execução no cálculo da métrica AUC (`roc_auc_score(..., multi_class='ovo')`), emitindo a exceção: *"Number of classes in y_true not equal to the number of columns in y_score"*. O cálculo pareado da Curva ROC torna-se matematicamente indefinido ao tentar mensurar a taxa de falsos positivos para classes sem instâncias amostrais no conjunto de avaliação.
- **Solução Metodológica (Filtragem Dinâmica):** Foi desenvolvida a função `compute_auc_safe` para tratar essa exceção dimensional. O método identifica as classes efetivamente presentes no conjunto de teste, aplica um recorte vetorial na matriz de probabilidades e realiza a re-normalização estatística. Essa abordagem garantiu a estabilidade do pipeline perante partições esparsas.

### B. Fallback para "Single Class" e NaNs Esperados
- **Problema:** Datasets como o `JapaneseVowels` demonstraram comportamento anômalo interno nos modelos base do AutoGluon devido ao formato intrínseco dos dados (timeseries multivariável forçada para tabular). Isso resultou em `AUC=nan`.
- **A Solução:** Adotamos o retorno explícito e gracioso de `np.nan` nestes casos matematicamente insolúveis, para preservar a integridade da extração de outros indicadores como Acurácia e Tempo de Treinamento, impedindo que o script encerrasse de forma prematura.

## 5. Arquitetura da Rodada Definitiva (Escalonamento e Limite de 32 CPUs no Apuana)
Para viabilizar a geração dos resultados finais e acomodar o custo computacional exigido pelos 10 maiores datasets do benchmark, adotou-se uma estratégia de maximização de paralelismo.

- **A Constatação:** O gargalo no tempo total não é linear. O maior tempo de espera vem do *AutoGluon Extreme*, que obedece a um teto fixo de horas. Se deixarmos o limite em 2h usando 8 CPUs, entregamos **16 horas-CPU** de poder de processamento interno.
- **A Decisão Final:** Criamos os scripts finais (`run_cluster_final.py` e `job_apuana_final.slurm`) solicitando originalmente **64 CPUs** e reduzindo o teto de tempo para **1 hora** para acelerar a validação. 
- **Teto Físico do Servidor:** Durante a submissão ao cluster Apuana, a alocação foi restringida pelas políticas de gestão de recursos da partição, impondo um limite de **32 CPUs**. A arquitetura operou sob esta restrição de infraestrutura, reduzindo o paralelismo projetado e estendendo marginalmente o tempo real de convergência das Árvores de Decisão nos datasets de grande porte.

O resultado esperado desta última arquitetura é a execução impecável de toda a fila de dados, finalizando o experimento em prazo acelerado e gerando métricas de ensemble com precisão superior.

## 6. Análise da Execução Final (Soft Limits vs Extensões C++)
Durante a execução de instâncias de alta complexidade (ex: `waveform-5000`), observou-se uma divergência no controle de tempo máximo (*timeout*), levantando um ponto metodológico relevante:

### A. O Comportamento Contraintuitivo dos Limites de Tempo
Observamos que o **AutoGluon Default**, configurado com um limite de 30 minutos (`time_limit=1800`), demorou **1 hora e 12 minutos** para concluir a predição, ultrapassando não apenas o seu próprio limite, mas também o tempo gasto pelo **AutoGluon Extreme** (cravado em 60 minutos exatos).

* **Por que isso aconteceu?** O limite nativo do AutoGluon é um *Soft Limit*. O framework avalia o cronômetro apenas no intervalo *entre* o treinamento de dois sub-modelos. Como o perfil *Default* obrigatoriamente engloba a tentativa de treino de Redes Neurais pesadas (FastAI/Torch), quando o algoritmo entra nessas redes no "minuto 29", ele fica travado processando as *epochs* e só vai verificar o relógio 40 minutos depois, estourando a barreira teórica de tempo.
* **Comportamento de Extensões C/C++:** Para assegurar o cumprimento do tempo de execução, foi implementado um controle de interrupção (via biblioteca `signal` do Python). Contudo, como o processamento primário (Árvores e Redes Neurais) é delegado a extensões compiladas em C++, os sinais do Python não interrompem a rotina subjacente até que o processo retorne o controle ao interpretador principal.

### B. A Inteligência do Perfil Extreme
Ironicamente, o modo *Extreme* (`best_quality`), com limite de 60 minutos, encerrou cravado no tempo correto (61 minutos). Isso ocorre porque, ao ativar o modo *best_quality*, o AutoGluon altera sua heurística interna: sabendo que tem pouco tempo para fazer dezenas de validações cruzadas (bagging e stacking), ele ativa uma auto-preservação e frequentemente decide **pular completamente as Redes Neurais lentas**, focando o tempo nas Árvores de Decisão rápidas para entregar um ensemble robusto dentro da regra do relógio.

### C. Análise Comparativa de Eficiência (TabICL)
Uma observação acadêmica substancial foi obtida no dataset `waveform-5000`. Enquanto os ensembles tradicionais do AutoGluon atingiram tempos de execução superiores a 1 hora, resultando em `AUC=0.9757` (Default) e `0.9744` (Extreme - denotando leve *overfitting* de ensemble), o modelo **TabICL** demonstrou alta otimização baseada em *In-Context Learning*. 
O modelo inferiu as predições em **5.6 segundos**, atingindo o maior valor absoluto para a amostra: **AUC=0.9785**. Tal resultado atesta não apenas precisão competitiva, mas ganho expressivo de eficiência computacional.

## 7. Curadoria de Datasets e o Paradoxo do Benchmark 10/10/10
Durante a fase final de consolidação dos 30 datasets exigidos pelo edital da disciplina, deparamo-nos com um impasse arquitetural significativo envolvendo as métricas do benchmark TabArena-v0.1 e as exigências do projeto.

### A. A Exigência Matemática do Edital
O edital determinava duas regras primárias: a utilização de **exatamente 30 datasets** da base TabArena e uma estratificação perfeita de **10 pequenos (<1000 amostras), 10 médios (1k-10k) e 10 grandes (>10k)**.

### B. A Realidade Curada (O Paradoxo da Classificação)
Ao auditar a planilha oficial de curadoria dos autores originais do TabArena (`TabArena_Dataset_Curation`), mapeamos uma limitação intrínseca da base que impossibilitava o cumprimento estrito das duas regras simultaneamente:
1. **Falta de Bases Pequenas:** Dos 50 datasets plenamente aprovados com o selo "Yes" de qualidade pelos curadores (Andrej e Lennart), **apenas 4 datasets se enquadravam como pequenos** (ex: `diabetes`, `ilpd`).
2. **Escassez de Bases Puras:** Quando aplicamos o filtro de restrições do nosso pipeline (exigir que a tarefa fosse estritamente de *Classificação* e que não sofresse de Data Leakage temporal ou de grupo), o universo de datasets perfeitos e aprovados caiu para **aproximadamente 18 datasets** no total.

### C. Solução e Compromisso Metodológico
Para atingir o requisito de 30 datasets sem comprometer o rigor científico (evitando o uso de bases de regressão ou submetidas a vazamento temporal), optou-se pela seguinte abordagem metodológica:
- Preenchemos as vagas restantes com datasets "clássicos" e estruturalmente sólidos do OpenML (como `mushroom` e `spambase`), mesmo sabendo que estes possuíam um *flag* de rejeição ou "condicional" pela curadoria moderna do TabArena por serem considerados fáceis ou obsoletos.
- **A Estratificação Final (3/17/10):** A escolha de usar 3 Pequenos, 17 Médios e 10 Grandes reflete esse funil de qualidade. Nós priorizamos a integridade do modelo (evitar regressão e leakage) em detrimento da simetria arbitrária de tamanhos.
- **Validação Docente:** Este paradoxo da base original foi documentado e validado em reunião com o professor, autorizando o uso da distribuição 3/17/10 como uma demonstração de rigor acadêmico frente às limitações do paper.
