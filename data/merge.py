import pandas as pd
import os

# Função para carregar e combinar as bases de dados e salvar em um novo arquivo CSV
def carregar_e_salvar_dataset(caminho_csv1="base.csv", caminho_csv2="base_negociacao.csv", caminho_saida="data/base_unificada.csv"):
    caminho_absoluto1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), caminho_csv1)
    caminho_absoluto2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), caminho_csv2)

    # Carregando os datasets
    termos_improprios = pd.read_csv(caminho_absoluto1)
    negociacoes = pd.read_csv(caminho_absoluto2)

    # Renomear colunas para garantir que os datasets tenham a mesma estrutura
    negociacoes = negociacoes.rename(columns={"texto": "text", "classificacao": "inappropriate"})

    # Combinar os datasets
    dataset_combinado = pd.concat([termos_improprios, negociacoes], ignore_index=True)

    # Salvar o dataset unificado em um novo arquivo CSV
    dataset_combinado.to_csv(caminho_saida, index=False)
    print(f"Base unificada salva em: {caminho_saida}")

# Exemplo de uso
carregar_e_salvar_dataset()
