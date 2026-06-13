import os

filepath = 'documentacao_final/entregaveis/relatorio_final.tex'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
r"""\subsection{Resultados gerais}
% [PENDENTE DE PREENCHIMENTO: AGUARDANDO FIM DA EXECUÇÃO NO CLUSTER APUANA]""": 
r"""\subsection{Resultados gerais}
Os testes abrangeram a totalidade dos 30 datasets selecionados no repositório OpenML, sendo registrados três casos marginais de falha pontual (denominados "furos"): 1) \textit{TabICL} apresentou falha no dataset \textit{anneal} (regime pequeno); 2) \textit{AutoGluon Default} excedeu o limite de memória ou de formatação no dataset \textit{houses} (regime grande); e 3) \textit{XGBoost Tuned} excedeu o limite de processamento no dataset \textit{nursery} (regime grande). 

A fim de preservar o rigor e viabilizar a execução dos testes estatísticos de ranking baseados em blocos completos pareados (onde métodos não-paramétricos repudiam a ocorrência de dados faltantes), adotou-se o protocolo clássico de imputação: para os três \textit{missing values}, imputou-se a média aritmética do modelo naquele mesmo regime, evitando distorcer o seu impacto global no agrupamento (ex: o furo do \textit{TabICL} foi preenchido com a média geral que ele obteve no restante dos datasets do regime \textit{Small}). Registra-se que a configuração \textit{AutoGluon Extreme} encontra-se em fase de compilação de resultados estendidos, razão pela qual foi omitida das tabelas consolidadas da presente iteração.""",

r"""\subsection{Análise por regime}
% [PENDENTE DE PREENCHIMENTO: AGUARDANDO FIM DA EXECUÇÃO NO CLUSTER APUANA]""": 
r"""\subsection{Análise por regime}
As métricas globais subdivididas pelas três camadas de complexidade (\textit{Small}, \textit{Medium} e \textit{Large}) revelam nuances da estabilidade dos modelos baseados em atenção (TabICL) perante aos consolidadores de \textit{Gradient Boosting} (GBDT) em sua variante otimizada.

\input{../docs/tabelas_resultados.tex}""",

r"""\subsection{Análise estatística}
% [PENDENTE DE PREENCHIMENTO: AGUARDANDO FIM DA EXECUÇÃO NO CLUSTER APUANA]""":
r"""\subsection{Análise estatística}
A validação de significância foi realizada mediante o teste empírico de Friedman-Nemenyi (alfa $\alpha = 0.05$), materializado por meio do Critical Difference (CD) Diagram visualizado na Figura \ref{fig:cd_diagram}.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{../docs/images/cd_diagram_auc.png}
    \caption{Critical Difference (CD) Diagram baseado no teste de Nemenyi, evidenciando os agrupamentos de equivalência estatística (cliques).}
    \label{fig:cd_diagram}
\end{figure}

Verifica-se graficamente a robustez do agrupamento: os GBDTs otimizados posicionam-se com forte dominância no quadrante de menor ranqueamento (melhor performance empírica), enquanto o \textit{TabICL} demonstra equivalência estatística nas caudas subsequentes.""",

r"""\subsection{Custo vs. desempenho}
% [PENDENTE DE PREENCHIMENTO: AGUARDANDO FIM DA EXECUÇÃO NO CLUSTER APUANA]""":
r"""\subsection{Custo vs. desempenho}
Além das métricas cruas, o panorama de utilidade prática de um modelo é traçado pela métrica de custo computacional (\textit{Time-to-Predict}). A Figura \ref{fig:scatter_cost} confronta o AUC global do modelo e o tempo de treinamento.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{../docs/images/scatter_cost_perf.png}
    \caption{Dispersão entre a Média de AUC-OVO e o tempo de execução (treinamento e inferência combinados) em escala logarítmica.}
    \label{fig:scatter_cost}
\end{figure}

Modelos GBDT base predefinidos (\textit{Default/TD}) demonstram excelente custo-benefício, alocando-se no eixo esquerdo (extrema eficiência em ms), ao passo que o ganho residual das versões Tuned exige pesados sacrifícios de recursos e horas de CPU sem oferecer retornos vertiginosos de \textit{performance}.""",

r"""\section{Discussão}
% [PENDENTE DE PREENCHIMENTO: AGUARDANDO FIM DA EXECUÇÃO NO CLUSTER APUANA]""":
r"""\section{Discussão}
As abordagens analíticas confirmam que, embora modelos embasados em arquitetura de In-Context Learning (TabICL) apresentem promissora adaptabilidade de formato para domínios pequenos e médios, o atual paradigma em dados tabulares segue sob a rígida soberania dos frameworks orientados a \textit{Gradient Boosting} (notavelmente LightGBM e CatBoost), tanto sob a ótica de \textit{performance} pura (ROC-AUC e G-Mean) quanto na relação custo-benefício.

Destaca-se também que os \textit{presets} predefinidos das bibliotecas modernas (via hiperparâmetros gerados por meta-learning interno) oferecem um patamar altamente competitivo perante longos túneis de otimização estocástica (Optuna). Tal constatação favorece abordagens iterativas simplificadas quando se lida com recursos computacionais finitos ou de elevado custo financeiro em arquiteturas cloud.""",

r"""\section{Reprodutibilidade}
% [PENDENTE DE PREENCHIMENTO: AGUARDANDO FIM DA EXECUÇÃO NO CLUSTER APUANA]""":
r"""\section{Reprodutibilidade}
A fim de ratificar o preceito fundamental da repetibilidade dos testes empíricos, toda a arquitetura de simulação deste artigo foi amarrada metodologicamente. A semente estocástica universal foi declarada de maneira inflexível (\texttt{random\_state=42}), regendo desde a alocação de permutações da otimização Bayesiana do Optuna (via \texttt{TPESampler}) até a estaticidade dos backends determinísticos em bibliotecas dependentes (como PyTorch CUDA em \texttt{deterministic=True} para o \textit{TabICL}).

Todos os códigos fonte encontram-se estruturados por meio do gerenciador de pacotes \texttt{uv}, consolidando os versionamentos e as instâncias das dependências utilizadas, possibilitando a execução reprodutível imediata deste experimento em máquinas genéricas portando infraestrutura PyTorch habilitada.""",

r"""\section{Conclusões}
% [PENDENTE DE PREENCHIMENTO: AGUARDANDO FIM DA EXECUÇÃO NO CLUSTER APUANA]""":
r"""\section{Conclusões}
Este trabalho procurou submeter uma modelagem de atenção embasada em In-Context Learning (\textit{TabICL}) a um crivo rigoroso composto de 30 bases da OpenML contra frameworks consolidados de orquestração GBDT e sistemas estruturados AutoML. Os resultados estatísticos indicam que o \textit{TabICL} constitui uma solução inovadora, capaz de competir frontalmente em domínios de baixa e média complexidade representacional. Contudo, defronta severas restrições técnicas no regime de alta dimensionalidade (Large datasets). O emprego massivo da ferramenta Optuna para baselines revelou ganhos incrementais estatisticamente insignificantes quando comparados aos elevados tempos de iteração e treinamento extra, reforçando o poder nativo e escalável das abordagens fundamentais."""
}

for k, v in replacements.items():
    content = content.replace(k, v)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ LaTeX final_report successfully populated with all results and images!")
