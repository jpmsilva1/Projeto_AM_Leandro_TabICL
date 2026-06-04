# Estudo de Viabilidade e Plano de Implementação: Migração para o Cluster Apuana (UFPE)

O objetivo deste documento é analisar a viabilidade de rodar o pipeline de avaliação do TabArena (TabICL v2 vs Baselines) no cluster computacional Apuana do Centro de Informática (CIn) da UFPE, além de fornecer um roteiro técnico detalhado para a execução.

## 1. Análise de Viabilidade

**Veredito: Altamente Viável e Recomendado.**

Rodar o projeto no cluster Apuana resolve os dois maiores gargalos que estamos enfrentando atualmente no Kaggle:

1. **Gargalo de VRAM (Erro OOM do TabICL):** No Kaggle, vimos que o TabICL exigiu ~25.6 GB de VRAM para rodar o dataset `APSFailure`, o que causou um erro na GPU Tesla T4 (que só tem 15.6 GB). Clusters institucionais como o Apuana geralmente são equipados com GPUs de alta capacidade (como NVIDIA A100 de 40GB/80GB ou RTX 3090/4090 de 24GB). Isso permitirá rodar o TabICL nos datasets grandes (Large Regime) sem estourar a memória.
2. **Gargalo de Tempo Limite (Sessão de 12 horas):** O Kaggle encerra execuções interativas após 12 horas. Com o SLURM (gerenciador do cluster Apuana), você pode submeter o seu script como um "job" em background. O código pode ficar rodando por 24, 48 ou 72 horas ininterruptas. Isso permite habilitar com segurança a opção **"extreme" do AutoGluon** para o projeto inteiro, sem perder noites de sono cuidando do terminal.

> [!WARNING]
> Restrições do Cluster: A documentação informa que o uso de Docker/Máquinas Virtuais é proibido por motivos de segurança. Portanto, teremos que configurar as dependências "na mão" através de módulos e ambientes virtuais Python, o que é um processo muito tranquilo.

---

## 2. Open Questions (Para o Usuário)

> [!IMPORTANT]
> - Você já possui conta/login no domínio `@cin.ufpe.br`?
> - Você já preencheu o formulário de acesso ao cluster e tem a VPN do CIn instalada na sua máquina?
> - O seu colega de grupo também usou esse cluster ou apenas o Kaggle? (Saber disso nos ajuda a entender se as bibliotecas como o `tabicl` já estão cacheadas nos nós do servidor).

---

## 3. Plano de Implementação Passo a Passo

O processo será dividido nas seguintes etapas que nós executaremos juntos assim que você tiver acesso:

### Etapa 1: Acesso e Configuração Inicial
1. Conectar na VPN do CIn.
2. Acessar o "Login Node" do cluster via SSH:
   ```bash
   ssh seu_login@slurm-client1.cin.ufpe.br
   ```
3. Fazer o clone do seu repositório Git com os scripts que nós já atualizamos:
   ```bash
   git clone https://github.com/jpmsilva1/Projeto_AM_Leandro_TabICL.git
   cd Projeto_AM_Leandro_TabICL
   ```

### Etapa 2: Criação do Ambiente Virtual
Segundo a documentação do Apuana, ativaremos os módulos Python do sistema e isolaremos as bibliotecas:
```bash
# Carrega a versão mais moderna de Python do servidor
module load Python3.10

# Cria um ambiente isolado na sua pasta Home
python -m venv $HOME/tabarena_env
source $HOME/tabarena_env/bin/activate

# Instala as dependências (semelhante à Célula 1 do Kaggle)
pip install tabicl "pytabkit[models]" openml optuna scikit-learn lightgbm xgboost catboost "autogluon.tabular[all]"
```

### Etapa 3: Adaptação do Código (Transformar o Notebook em Script)
Nós transformaremos as "Células" do arquivo `kaggle_pipeline.md` em um único script Python limpo (ex: `run_cluster.py`). 
- Iremos reativar a configuração `presets="extreme_quality"` do AutoGluon.
- Removeremos a trava de `BATCH_START` e `BATCH_END`, para que ele processe todos os 16 datasets de uma só vez.

### Etapa 4: Submissão via SLURM (sbatch)
Em vez de rodar o código de forma interativa, criaremos um arquivo chamado `job.slurm`. Este arquivo avisa ao cluster que precisamos de uma GPU e memória RAM alocadas:

```bash
#!/bin/bash
#SBATCH --job-name=TabICL_Eval
#SBATCH --output=resultado_tabarena.txt
#SBATCH --error=erros_tabarena.txt
#SBATCH --time=48:00:00        # Pede até 48 horas de tempo
#SBATCH --gpus=1               # Pede 1 placa de vídeo
#SBATCH --cpus-per-task=8      # Pede 8 processadores 

module load Python3.10
source $HOME/tabarena_env/bin/activate
python run_cluster.py
```

Você fará a submissão rodando apenas `sbatch job.slurm`. A partir desse momento, você pode desligar o seu computador e ir dormir. O servidor da UFPE fará o trabalho e salvará os CSVs.

### Etapa 5: Resgate dos Resultados
No dia seguinte, usaremos o comando `scp` (Secure Copy) ou o VSCode para baixar o arquivo `kaggle_results.csv` e `dataset_metadata.csv` do cluster diretamente para o seu Mac, e finalizamos as análises estatísticas!
