#!/bin/bash
# ===================================================================
# TATICA ENXAME — SCRIPT DE SETUP AUTOMATICO
# Execute este script UMA UNICA VEZ ao acessar o cluster pela 1a vez.
#
# USO:
#   chmod +x setup_enxame.sh
#   ./setup_enxame.sh
#
# Ele faz TUDO sozinho: clona o repo, cria o ambiente, instala libs.
# ===================================================================

set -e  # Para imediatamente se qualquer comando falhar

echo "=========================================="
echo "  TATICA ENXAME - Setup Automatico"
echo "  $(date)"
echo "=========================================="

# 1. Clonar o repositorio (se ainda nao existe)
REPO_DIR="$HOME/Projeto_AM_Leandro_TabICL"
if [ -d "$REPO_DIR" ]; then
    echo "[OK] Repositorio ja existe em $REPO_DIR"
    cd "$REPO_DIR"
    git pull origin main
else
    echo "[...] Clonando repositorio..."
    git clone https://github.com/jpmsilva1/Projeto_AM_Leandro_TabICL.git "$REPO_DIR"
    cd "$REPO_DIR"
fi

echo "[OK] Repositorio atualizado."

# 2. Criar o ambiente virtual Python (se nao existe)
VENV_DIR="$HOME/tabarena_env"
if [ -d "$VENV_DIR" ]; then
    echo "[OK] Ambiente virtual ja existe em $VENV_DIR"
else
    echo "[...] Criando ambiente virtual Python..."
    python3 -m venv "$VENV_DIR"
    echo "[OK] Ambiente virtual criado."
fi

# 3. Ativar o ambiente
source "$VENV_DIR/bin/activate"
echo "[OK] Ambiente ativado: $(which python3)"

# 4. Instalar dependencias (apenas o necessario para o AutoGluon)
echo "[...] Instalando dependencias (isso pode demorar 10-15 min na primeira vez)..."
pip install --upgrade pip -q
pip install autogluon openml pandas numpy scikit-learn -q
echo "[OK] Dependencias instaladas."

# 5. Testar se tudo funciona
echo "[...] Testando imports..."
python3 -c "from autogluon.tabular import TabularPredictor; print('[OK] AutoGluon funcionando!')"
python3 -c "import openml; print('[OK] OpenML funcionando!')"
python3 -c "import sklearn; print('[OK] Scikit-learn funcionando!')"

# 6. Criar pasta de logs
mkdir -p "$REPO_DIR/cluster_apuana/logs"
echo "[OK] Pasta de logs criada."

# 7. Verificacao final
echo ""
echo "=========================================="
echo "  SETUP CONCLUIDO COM SUCESSO!"
echo "=========================================="
echo ""
echo "  Agora execute:"
echo "    cd ~/Projeto_AM_Leandro_TabICL/cluster_apuana"
echo "    sbatch job_ag_SEUNOME.slurm"
echo ""
echo "  Substitua SEUNOME pelo seu nome:"
echo "    Clara   -> sbatch job_ag_clara.slurm"
echo "    Vinicius -> sbatch job_ag_vinicius.slurm"
echo ""
echo "=========================================="
