# 🐝 Tática Enxame — Guia Completo para Clara e Vinícius

> **Missão:** Rodar o AutoGluon Extreme com **4 horas de limite** por dataset, dividindo 30 datasets entre 3 pessoas para terminar tudo até **sábado**.

> [!IMPORTANT]
> O João já está rodando a parte dele (10 datasets LARGE) desde a madrugada de quinta/sexta. Este guia é para a **Clara** e o **Vinícius** configurarem e rodarem a parte deles.

---

## Quem Roda O Quê

| Membro | Datasets | Dificuldade | Tempo Estimado | Início |
|---|---|---|---|---|
| **João** (`jpms5`) | 10 LARGE | 🔴 Pesados | ~35–40h | ✅ Já rodando! |
| **Clara** | 10 MEDIUM | 🟡 Médios | ~20–25h | Sexta de manhã |
| **Vinícius** | 10 SMALL/MEDIUM | 🟢 Leves | ~10–15h | Sexta (manhã ou noite) |

---

## ETAPA 1 — Abrir o Terminal no seu Computador

Antes de acessar o cluster, você precisa abrir o terminal do seu computador (Windows).

### Windows 10:
1. Clique no **Menu Iniciar** (ícone do Windows no canto inferior esquerdo)
2. Digite **`cmd`** ou **`PowerShell`**
3. Clique em **"Prompt de Comando"** ou **"Windows PowerShell"**
4. Uma janela preta (ou azul) vai abrir com um cursor piscando. **Este é o seu terminal.**

### Windows 11:
1. Clique com o **botão direito** no Menu Iniciar
2. Clique em **"Terminal"**
3. Uma janela vai abrir com um cursor piscando. **Este é o seu terminal.**

> [!NOTE]
> Se por acaso aparecer o erro `'ssh' não é reconhecido como um comando interno`, rode isso no PowerShell como Administrador:
> ```powershell
> Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
> ```
> Depois feche e abra o terminal novamente.

---

## ETAPA 2 — Conectar na VPN do CIn (OBRIGATÓRIO ANTES DO SSH!)

O cluster Apuana fica dentro da rede interna da universidade. Para acessá-lo de casa, você **precisa primeiro se conectar à VPN do CIn**. Sem a VPN, o comando SSH simplesmente não vai funcionar (vai ficar "travado" sem responder).

> [!CAUTION]
> **A conexão VPN dura apenas 30 minutos!** Depois disso, ela cai automaticamente. Porém, **isso NÃO afeta o seu job no cluster** — ele continua rodando sozinho. Você só precisa reconectar a VPN + SSH se quiser verificar o progresso novamente.

### 2.1 Baixar o cliente VPN (primeira vez)

Se você ainda não tem o cliente VPN instalado:
1. Acesse o portal de VPN do CIn pelo navegador (pergunte ao João o link exato se não souber)
2. Baixe o cliente **OpenVPN** ou **FortiClient** (depende de qual a universidade usa)
3. Instale normalmente no Windows (Next → Next → Finish)

### 2.2 Conectar à VPN

1. Abra o programa da VPN (procure por "OpenVPN" ou "FortiClient" no Menu Iniciar)
2. Coloque o **endereço do servidor VPN**, seu **usuário** e **senha** do CIn
3. Clique em **Conectar**
4. Espere até aparecer **"Conectado"** (ou um ícone verde na barra de tarefas)

### 2.3 O timer de 30 minutos

A partir do momento que você conectou na VPN, você tem **~30 minutos** antes dela cair.

**O que fazer:**
- **Para submeter o job (Etapas 3–4):** 30 minutos é mais que suficiente. Você só precisa da VPN para digitar os comandos de setup e o `sbatch`. Depois disso, pode desconectar à vontade.
- **Para verificar o progresso (Etapa 5):** Reconecte a VPN, faça SSH, rode o `squeue` ou `tail -f`, veja o que precisa e pronto.

> [!IMPORTANT]
> **Fluxo correto toda vez que for acessar o cluster:**
> 1. Conectar na VPN do CIn ← **sempre primeiro!**
> 2. Abrir o terminal (PowerShell)
> 3. Fazer SSH (`ssh SEU_USUARIO@slurm-client1.cin.ufpe.br`)
> 
> Se a VPN cair no meio de uma sessão SSH, o terminal vai travar ou mostrar `Connection reset`. Basta reconectar a VPN e fazer SSH de novo. **O job no cluster NÃO é afetado.**

---

## ETAPA 3 — Conectar no Cluster Apuana via SSH

SSH é como um "controle remoto" que permite você controlar o computador do cluster (que está na universidade) a partir do seu computador de casa. **Certifique-se de que a VPN está conectada antes de prosseguir!**

> [!NOTE]
> Se você ainda não tem conta no cluster Apuana, solicite preenchendo este formulário: https://forms.gle/rU2te4TfvnjAdiqS8
> A aprovação pode demorar algumas horas. Fale com o João se precisar de urgência.

### 3.1 Digitar o comando de conexão

No terminal que você abriu, digite **exatamente** isso (substituindo `SEU_USUARIO` pelo seu login do cluster):

```bash
ssh SEU_USUARIO@slurm-client1.cin.ufpe.br
```

**Exemplo:** Se o seu login é `clara123`, você digita:
```bash
ssh clara123@slurm-client1.cin.ufpe.br
```

Aperte **Enter**.

### 3.2 O que vai aparecer na tela

**Na primeira vez**, vai aparecer esta pergunta:
```
The authenticity of host 'slurm-client1.cin.ufpe.br' can't be established.
ECDSA key fingerprint is SHA256:xxxxxxxxxxx
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

**Digite `yes`** (por extenso, as 3 letras) e aperte **Enter**.

### 3.3 Digitar a senha

Vai aparecer:
```
SEU_USUARIO@slurm-client1.cin.ufpe.br's password:
```

**⚠️ ATENÇÃO:** Quando você digitar a sua senha, **NADA vai aparecer na tela**. Nenhum asterisco, nenhum pontinho. A tela vai parecer travada. **Isso é NORMAL e é uma medida de segurança do Linux.** Apenas digite a senha normalmente e aperte **Enter**.

### 3.4 Confirmando que logou

Se a senha estiver correta, você verá algo como:
```
Last login: Fri Jun 12 08:00:00 2026 from 172.x.x.x
[SEU_USUARIO@slurm-client1:~]$
```

**Se viu esse `$` com cursor piscando, PARABÉNS! Você está dentro do cluster!** 🎉

> [!WARNING]
> Se aparecer `Permission denied, please try again`, a senha está errada. Tente novamente com calma. Após 3 tentativas erradas, o sistema pode bloquear por alguns minutos. Se não conseguir de jeito nenhum, fale com o João.

---

## ETAPA 4 — Rodar o Setup Automático (UMA ÚNICA VEZ)

Agora você precisa preparar o ambiente. Eu criei um script que faz **tudo sozinho**. Você só precisa copiar e colar.

### 4.1 Copiar e colar os comandos abaixo

**Selecione TUDO abaixo** (as 4 linhas), copie (Ctrl+C) e cole no terminal (clique com o botão direito no terminal do Windows para colar, ou Ctrl+Shift+V):

```bash
cd ~
git clone https://github.com/jpmsilva1/Projeto_AM_Leandro_TabICL.git
cd Projeto_AM_Leandro_TabICL/cluster_apuana/tatica_enxame
chmod +x setup_enxame.sh && ./setup_enxame.sh
```

### 4.2 O que você vai ver na tela

O terminal vai começar a cuspir muitas linhas. Isso é normal! Ele está:
1. Baixando o código do nosso projeto do GitHub
2. Criando um ambiente Python isolado
3. Instalando as bibliotecas (AutoGluon, OpenML, etc.)

```
[OK] Repositorio atualizado.
[...] Criando ambiente virtual Python...
[OK] Ambiente virtual criado.
[...] Instalando dependencias (isso pode demorar 10-15 min na primeira vez)...
```

> [!CAUTION]
> **NÃO FECHE O TERMINAL!** A instalação demora de **10 a 20 minutos**. Se você fechar, vai ter que começar do zero. Vá tomar um café e volte.

### 4.3 Confirmando que deu certo

Quando tudo terminar, você verá:
```
[OK] AutoGluon funcionando!
[OK] OpenML funcionando!
[OK] Scikit-learn funcionando!
[OK] Pasta de logs criada.

==========================================
  SETUP CONCLUIDO COM SUCESSO!
==========================================
```

**Se viu `SETUP CONCLUIDO COM SUCESSO!`, está tudo pronto!** ✅

Se deu algum erro, tire um print da tela e mande para o João no WhatsApp.

---

## ETAPA 5 — Submeter o Job (O MOMENTO DA VERDADE!)

Agora vamos mandar o cluster começar a rodar o nosso experimento. **São apenas 2 comandos.**

### Se você é a CLARA:
```bash
cd ~/Projeto_AM_Leandro_TabICL/cluster_apuana/tatica_enxame/clara
sbatch job.slurm
```

### Se você é o VINÍCIUS:
```bash
cd ~/Projeto_AM_Leandro_TabICL/cluster_apuana/tatica_enxame/vinicius
sbatch job.slurm
```

### O que vai aparecer:
```
Submitted batch job 3456
```

O número (`3456`) é o ID do seu job. **Anote este número!** Você vai precisar dele para verificar o progresso.

**🎉 PRONTO! O experimento está rodando!** A partir de agora, o computador do cluster vai trabalhar sozinho processando os 10 datasets automaticamente, um atrás do outro. Você NÃO precisa ficar na frente do computador. Pode fechar o terminal, desligar o PC, ir dormir — o job continua rodando lá no servidor da universidade.

---

## ETAPA 6 — Verificar o Progresso

Você pode (e deve!) verificar como está o progresso de vez em quando. Para isso:

1. **Reconecte a VPN do CIn** (lembre: ela expira a cada 30 min)
2. Abra o terminal e faça SSH novamente:

```bash
ssh SEU_USUARIO@slurm-client1.cin.ufpe.br
```

### 5.1 Comando: `squeue` — Ver se o job está rodando

```bash
squeue -u SEU_USUARIO
```

**O que vai aparecer:**
```
  JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
   3456 long-simp ag_clara SEU_USER  R    2:30:15      1 cluster-node3
```

**Como ler esta tabela:**

| Coluna | Significado |
|---|---|
| `JOBID` | O número do seu job (o mesmo que apareceu no `sbatch`) |
| `NAME` | O nome do job (`ag_clara` ou `ag_viniciu`) |
| `ST` | **Status** — a coluna mais importante! |
| `TIME` | Há quanto tempo está rodando |
| `NODELIST` | Em qual computador do cluster está rodando |

**Valores da coluna ST (Status):**

| Status | Significado | O que fazer |
|---|---|---|
| `R` | **Running** — Está rodando! 🟢 | Nada, apenas espere |
| `PD` | **Pending** — Na fila de espera ⏳ | Espere, pode levar alguns minutos |
| (vazio) | O job já terminou | Verifique os logs (próximo passo) |

### 5.2 Comando: `tail -f` — Ver o progresso AO VIVO

Este é o comando mais legal! Ele mostra em tempo real o que o Python está fazendo:

```bash
tail -f ~/Projeto_AM_Leandro_TabICL/cluster_apuana/logs/ag_extreme_*.out
```

**O que você vai ver (exemplo):**
```
2026-06-13 08:35:00 [INFO] =================================================================
2026-06-13 08:35:00 [INFO]   Dataset: spambase (tid=43, regime=medium)
2026-06-13 08:35:00 [INFO] =================================================================
2026-06-13 08:35:01 [INFO]   ⚙️  Iniciando AutoGluon Extreme (limite=4h / 14400s)...
```

E quando um dataset terminar:
```
2026-06-13 12:34:55 [INFO]   ✅ AutoGluon_Extreme_4h | ACC=0.9456 | AUC=0.9727 | Tempo=14372.3s
2026-06-13 12:34:55 [INFO]   💾 Resultado salvo em ag_extreme_clara.csv
```

**Para SAIR dessa tela:** aperte **Ctrl + C** (segure Ctrl e aperte C). **Isso NÃO cancela o job!** Só para de mostrar os logs na tela. O job continua rodando normalmente.

### 5.3 Comando: `cat` — Ver se houve erros

```bash
cat ~/Projeto_AM_Leandro_TabICL/cluster_apuana/logs/ag_extreme_*.err
```

- Se **não apareceu nada** (o terminal só pulou uma linha): ótimo, zero erros! ✅
- Se apareceu algum texto em vermelho ou mensagens de erro: tire um print e mande para o João.

### 5.4 Comando: `cat` — Ver os resultados que já foram salvos

```bash
cat ~/Projeto_AM_Leandro_TabICL/cluster_apuana/ag_extreme_clara.csv
```
(ou `ag_extreme_vinicius.csv` para o Vinícius)

Isso vai mostrar uma tabela CSV com os datasets que já foram processados, incluindo a acurácia, AUC, G-Mean, etc.

### 5.5 Contar quantos datasets já terminaram

```bash
wc -l ~/Projeto_AM_Leandro_TabICL/cluster_apuana/ag_extreme_clara.csv
```

O número mostrado **menos 1** (por causa do cabeçalho) é a quantidade de datasets concluídos. Se mostrar `6`, significa que 5 dos 10 datasets já terminaram.

---

## ETAPA 7 — Entregar os Resultados ao João

Quando todos os 10 datasets terminarem (você vai ver `BATCH FINALIZADO!` nos logs, ou o `wc -l` vai mostrar `11` = 10 datasets + 1 cabeçalho), faça o seguinte:

### Opção A — Via Git (Preferível):
```bash
cd ~/Projeto_AM_Leandro_TabICL/cluster_apuana/tatica_enxame/clara
git add ag_extreme_clara.csv
git commit -m "Resultados AG Extreme 4h - Clara"
git push origin main
```
(Substitua `clara` por `vinicius` se for o caso)

### Opção B — Copiar o conteúdo e mandar pelo WhatsApp:
```bash
cat ~/Projeto_AM_Leandro_TabICL/cluster_apuana/tatica_enxame/clara/ag_extreme_clara.csv
```
Selecione TODA a saída, copie e mande para o João no grupo do WhatsApp.

---

## Glossário — O que significam esses termos?

| Termo | Significado simples |
|---|---|
| **SSH** | "Controle remoto" para acessar o computador do cluster pela internet |
| **Cluster** | Um conjunto de computadores potentes na universidade que fazem cálculos pesados |
| **SLURM** | O "gerente de fila" do cluster — ele decide quem roda quando |
| **Job** | Uma tarefa que você mandou o cluster executar |
| **sbatch** | Comando para enviar um job para a fila do SLURM |
| **squeue** | Comando para ver os jobs na fila |
| **tail -f** | Comando para acompanhar um arquivo ao vivo (como ver TV) |
| **Ctrl + C** | Atalho para sair do `tail -f` (NÃO cancela o job!) |
| **PD (Pending)** | O job está esperando na fila por um computador livre |
| **R (Running)** | O job está rodando em um computador |
| **CSV** | Arquivo de tabela (pode ser aberto no Excel) |
| **AutoGluon** | Biblioteca de Machine Learning que testa vários modelos automaticamente |
| **Dataset** | Conjunto de dados que o modelo vai treinar e testar |

---

## Problemas Comuns e Soluções

### ❌ `Permission denied` ao fazer SSH
**Causa:** Senha errada.
**Solução:** Tente novamente com calma. Lembre: os caracteres NÃO aparecem. Se errar 3 vezes, espere 5 minutos e tente de novo.

### ❌ `ssh: Could not resolve hostname`
**Causa:** Erro de digitação no endereço ou sem internet.
**Solução:** Verifique se digitou `slurm-client1.cin.ufpe.br` corretamente. Teste sua internet.

### ❌ `QOSMaxCpuPerUserLimit` ao submeter com sbatch
**Causa:** Você já tem um job rodando ou está pedindo CPUs demais.
**Solução:** Verifique com `squeue -u SEU_USUARIO`. Se já tem job, espere ele terminar.

### ❌ `ModuleNotFoundError: No module named 'autogluon'`
**Causa:** O setup não foi executado ou falhou no meio.
**Solução:** Rode o setup novamente:
```bash
cd ~/Projeto_AM_Leandro_TabICL/cluster_apuana
./setup_enxame.sh
```

### ❌ O terminal fechou sem querer / internet caiu / PC desligou
**NÃO SE PREOCUPE!** O job continua rodando no cluster. Basta fazer SSH de novo e verificar:
```bash
ssh SEU_USUARIO@slurm-client1.cin.ufpe.br
squeue -u SEU_USUARIO
```

### ❌ `squeue` não mostra nada (o job sumiu!)
O job pode ter terminado (ótimo!) ou ter sido cancelado (ruim). Verifique os logs:
```bash
cat ~/Projeto_AM_Leandro_TabICL/cluster_apuana/logs/ag_extreme_*.out | tail -20
cat ~/Projeto_AM_Leandro_TabICL/cluster_apuana/logs/ag_extreme_*.err
```
Se no `.out` aparece `BATCH FINALIZADO!`, terminou com sucesso! Se no `.err` aparece algum erro, mande o print para o João.

### ❌ Job foi cancelado (OOM Kill / memória insuficiente)
O script salva o progresso a cada dataset terminado. Basta resubmeter:
```bash
cd ~/Projeto_AM_Leandro_TabICL/cluster_apuana/tatica_enxame/clara
sbatch job.slurm
```
Ele vai **pular automaticamente** os datasets que já foram concluídos e continuar de onde parou. Nenhum progresso é perdido!

### ❌ `fatal: destination path already exists`
Você já clonou o repositório antes. Apenas atualize:
```bash
cd ~/Projeto_AM_Leandro_TabICL
git pull origin main
```

---

## Cronograma Previsto

### Se Clara começar sexta de manhã (8h):
| Horário | O que acontece |
|---|---|
| 08:00 | Faz SSH, roda o setup (~20min) |
| 08:25 | `sbatch job_ag_clara.slurm` — job submetido |
| 08:30 | Job começa a rodar o 1º dataset (hiva_agnostic) |
| ~11:00 | 1º dataset concluído, começa o 2º (spambase) |
| ~14:00 | 2º dataset concluído, começa o 3º... |
| ~22:00–00:00 | **Todos os 10 datasets finalizados!** 🎉 |

### Se Vinícius começar sexta à noite (20h):
| Horário | O que acontece |
|---|---|
| 20:00 | Faz SSH, roda o setup (~20min) |
| 20:25 | `sbatch job_ag_vinicius.slurm` — job submetido |
| 20:30 | Job começa a rodar o 1º dataset (blood-transfusion) |
| ~21:30 | 1º dataset concluído rápido (é pequeno!), começa o 2º... |
| Sábado ~08:00–10:00 | **Todos os 10 datasets finalizados!** 🎉 |

---

## Resumo Ultra-Rápido (Cola)

```
1. Conectar na VPN do CIn (SEMPRE PRIMEIRO!)
2. Abrir terminal (PowerShell)
3. ssh SEU_USUARIO@slurm-client1.cin.ufpe.br
4. (primeira vez) Rodar o setup:
     git clone https://github.com/jpmsilva1/Projeto_AM_Leandro_TabICL.git
     cd Projeto_AM_Leandro_TabICL/cluster_apuana/tatica_enxame
     chmod +x setup_enxame.sh && ./setup_enxame.sh
5. Submeter (entrar na SUA pasta):
     cd ~/Projeto_AM_Leandro_TabICL/cluster_apuana/tatica_enxame/clara
     sbatch job.slurm
6. Verificar (reconectar VPN + SSH primeiro):
     squeue -u SEU_USUARIO        (ver se está rodando)
     tail -f logs/ag_extreme_*.out  (ver ao vivo)
     Ctrl+C para sair do tail
7. Quando terminar, mandar CSV para o João
```

> [!TIP]
> **Lembrete final:** Esta operação é um bônus. O João já tem um pipeline principal rodando que garante a entrega do TCC. A Tática Enxame busca o resultado perfeito com 4 horas de AutoGluon. Se algo der errado, a tabela base já está garantida!
