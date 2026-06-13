FROM python:3.11-slim

# Evita geracao de pyc e garante output nao-bufferizado
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instala ferramentas essenciais de build do Linux e Curl para instalar o UV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Instala o gerenciador ultra-rápido de pacotes UV
RUN pip install --no-cache-dir uv

# Define o diretório base dentro do container
WORKDIR /app

# Copia os manifestos de pacote do projeto (incluindo o uv.lock para reprodutibilidade exata)
COPY pyproject.toml uv.lock ./

# Sincroniza o ambiente instalando as exatas versões blindadas pelo uv.lock
RUN uv sync --frozen

# Copia apenas a pasta contendo os scripts vitais da entrega
COPY src/ ./src/

# Configura o ponto de entrada usando o ambiente isolado do UV para rodar o pipeline
CMD ["uv", "run", "python", "src/run_cluster_final_v2.py"]
