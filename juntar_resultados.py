import os
import glob
import pandas as pd

def main():
    print("Iniciando a concatenação dos arquivos do Enxame (João, Vinicius, Clara)...")
    
    # Define explicitamente os arquivos para não poluir o dataframe final com arquivos auxiliares
    arquivos_csv = [
        "resultados_joao.csv", 
        "ag_extreme_joao.csv", 
        "ag_extreme_vinicius.csv", 
        "ag_extreme_clara.csv"
    ]
    
    if not arquivos_csv:
        print("Erro: Nenhum arquivo .csv encontrado nesta pasta para fazer o merge!")
        print("Por favor, coloque os CSVs do João, Vinicius e Clara aqui na mesma pasta deste script.")
        return

    print(f"Arquivos encontrados para mesclar: {arquivos_csv}")
    
    dataframes = []
    for arquivo in arquivos_csv:
        print(f"  Lendo: {arquivo}...")
        df = pd.read_csv(arquivo)
        dataframes.append(df)
        
    # Concatena todos os arquivos um embaixo do outro
    df_consolidado = pd.concat(dataframes, ignore_index=True)
    
    # Opcional: remove eventuais linhas duplicadas se alguém rodou o mesmo dataset sem querer
    tamanho_antes = len(df_consolidado)
    df_consolidado.drop_duplicates(subset=['dataset', 'model'], inplace=True)
    tamanho_depois = len(df_consolidado)
    
    if tamanho_antes != tamanho_depois:
        print(f"Aviso: Foram removidas {tamanho_antes - tamanho_depois} linhas duplicadas (mesmo modelo no mesmo dataset).")
        
    # Salva no caminho exato onde o script gerador de gráficos espera encontrar
    caminho_destino = "cluster_apuana/resultados_estat_finais/data/final_run_results_v2.csv"
    os.makedirs(os.path.dirname(caminho_destino), exist_ok=True)
    
    df_consolidado.to_csv(caminho_destino, index=False)
    
    print("\n✅ SUCESSO!")
    print(f"O arquivo unificado com {tamanho_depois} linhas foi salvo em:")
    print(f"-> {caminho_destino}")
    print("Agora você já pode rodar o comando: uv run python cluster_apuana/resultados_estat_finais/scripts/gerar_graficos_e_tabelas.py")

if __name__ == "__main__":
    main()
