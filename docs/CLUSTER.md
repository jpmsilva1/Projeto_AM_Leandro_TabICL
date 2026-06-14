# 🚀 Guia de Sobrevivência: Comandos Apuana

Este documento centraliza todos os comandos que você precisa para controlar a rodada final do seu projeto.

---

## 💻 Cluster Apuana (CIn-UFPE)
O Apuana utiliza o sistema de filas **SLURM**. O seu projeto está dentro da pasta `cluster_apuana`.

### 🟢 Iniciar os Trabalhos
Sempre que quiser rodar a versão mais nova do código ou reiniciar, faça este combo:
```bash
# 1. Entrar na pasta do projeto
cd ~/Projeto_AM_Leandro_TabICL/cluster_apuana

# 2. Puxar as atualizações do código (Correções de bugs)
git pull

# 3. Colocar o Job Final na fila do supercomputador (Forçando 32 CPUs)
sbatch --cpus-per-task=32 job_apuana_final.slurm
```

### 🔎 Monitorar o Progresso (O Famoso "Espiar")
Se você acabou de abrir o terminal SSH do CIn e quer ver como estão as coisas:
```bash
# 1. Entrar na pasta (sempre necessário em um novo terminal)
cd ~/Projeto_AM_Leandro_TabICL/cluster_apuana

# 2. Assistir ao log do Job Principal ao vivo
tail -f resultado_final_run.txt
```
*(Para sair da tela do `tail`, aperte `Ctrl + C`)*

### 🛑 Gerenciamento e Cancelamento
```bash
# Ver a lista dos seus Jobs rodando ou na fila (para descobrir o ID deles)
squeue -u jpms5

# Cancelar um Job específico (troque 1234 pelo ID que aparece no squeue)
scancel 1234
```

### 📥 Resgate dos Resultados
Quando o job finalizar amanhã, baixe o CSV com os resultados consolidados para o seu Mac:
```bash
# Rode isso em uma nova aba do terminal do seu Mac (NÃO DENTRO do Apuana):
scp jpms5@slurm-client1.cin.ufpe.br:~/Projeto_AM_Leandro_TabICL/cluster_apuana/final_run_results.csv ~/Downloads/
```
