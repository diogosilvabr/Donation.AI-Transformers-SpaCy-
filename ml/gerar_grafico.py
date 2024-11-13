import pandas as pd
import matplotlib.pyplot as plt

# Função para salvar e organizar métricas em CSV, mantendo o histórico e ajustando as épocas
def salvar_metricas_csv(metrics_df, caminho_csv="./runs/metrics_acumuladas.csv"):
    try:
        # Carrega dados anteriores, se existirem
        historico_df = pd.read_csv(caminho_csv)
        
        # Identifica a última época registrada e ajusta o índice das novas métricas
        ultima_epoca = historico_df['epoch'].max() if 'epoch' in historico_df.columns else 0
        metrics_df['epoch'] += ultima_epoca  # Ajusta para que a nova execução continue de onde parou

        # Concatena os dados antigos com os novos
        metrics_df = pd.concat([historico_df, metrics_df], ignore_index=True)
    except FileNotFoundError:
        print("Nenhum histórico encontrado. Criando novo arquivo de métricas.")
    finally:
        # Salva o dataframe atualizado em CSV
        metrics_df.to_csv(caminho_csv, index=False)

# Função para plotar o gráfico de época x acurácia e salvar como imagem
def plotar_grafico_acuracia_novo(caminho_csv="./runs/metrics_acumuladas.csv", caminho_imagem="./runs/epoch_vs_accuracy_acumulado.png"):
    # Carrega o CSV com a estrutura atual
    df_metrics = pd.read_csv(caminho_csv)

    # Verifica se a coluna `eval_accuracy` ou `accuracy` está presente
    if 'eval_accuracy' in df_metrics.columns:
        metric_col = 'eval_accuracy'
    elif 'accuracy' in df_metrics.columns:
        metric_col = 'accuracy'
    else:
        raise KeyError("Nenhuma coluna de acurácia ('eval_accuracy' ou 'accuracy') encontrada no arquivo.")

    # Ordena o DataFrame pela coluna 'epoch' para garantir a continuidade no gráfico
    df_metrics = df_metrics.sort_values(by="epoch")

    # Verifica se há dados para plotar
    if not df_metrics.empty:
        df_metrics.plot(x="epoch", y=metric_col, kind="line", title="Épocas x Acurácia - Validação")
        plt.xlabel("Épocas")
        plt.ylabel("Acurácia")
        plt.savefig(caminho_imagem)
        plt.show()
    else:
        print("Não há dados de 'eval_accuracy' ou 'accuracy' para gerar o gráfico.")

# Exemplo de chamada da função para salvar métricas
# Exemplo de dados (substitua pelo DataFrame real gerado no treinamento)
# metrics_df = pd.DataFrame(trainer.state.log_history)  # Por exemplo, após o treinamento
# salvar_metricas_csv(metrics_df)

# Executa a função para gerar o gráfico
plotar_grafico_acuracia_novo(caminho_csv="./runs/metrics_acumuladas.csv")
