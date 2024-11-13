import pandas as pd
import matplotlib.pyplot as plt
from transformers import (
    BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
)
from datasets import load_dataset

# Função para calcular a acurácia como métrica
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    accuracy = (predictions == labels).astype(float).mean().item()
    return {"accuracy": accuracy}

# Função para carregar e tokenizar o dataset
def preparar_dados(caminho_csv="./data/base_unificada.csv"):
    dataset = load_dataset("csv", data_files=caminho_csv)['train']
    dataset = dataset.rename_column("inappropriate", "labels").train_test_split(test_size=0.2, seed=42)
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    return dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=60), batched=True)

# Função para salvar métricas no CSV, preservando dados antigos e organizando as épocas
def salvar_metricas_csv(metrics_df, caminho_csv="./runs/metrics_acumuladas.csv"):
    try:
        # Carrega dados anteriores, se existirem
        historico_df = pd.read_csv(caminho_csv)
        # Concatena os dados antigos com os novos, mantendo o histórico
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

    # Verifica se a coluna `eval_accuracy` está presente
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
        plt.xticks(range(0, 31, 5))  # Marca as épocas de 5 em 5 até 30
        plt.savefig(caminho_imagem)
        plt.show()
    else:
        print("Não há dados de 'eval_accuracy' ou 'accuracy' para gerar o gráfico.")

# Configura o treinamento, executa o fine-tuning e salva as métricas
def executar_fine_tuning():
    modelo = BertForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased", num_labels=2)
    dataset = preparar_dados()
    
    trainer = Trainer(
        model=modelo,
        args=TrainingArguments(
            output_dir="./runs",
            evaluation_strategy="epoch",  # Avaliação a cada época
            save_strategy="epoch",  # Salvamento do modelo a cada época
            logging_dir="./runs/logs",
            num_train_epochs=30,  # Configurado para 30 épocas
            per_device_train_batch_size=8,  # Ajuste conforme sua capacidade de memória
            per_device_eval_batch_size=8,
            learning_rate=3e-5,  # Taxa de aprendizado adequada para treinamento longo
            weight_decay=0.01,
            load_best_model_at_end=True,
            logging_steps=50,  # Define a frequência de log detalhado
            save_total_limit=2  # Limite de checkpoints para evitar excesso de arquivos
        ),
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics
    )

    # Executa o treinamento e armazena as métricas em CSV
    train_result = trainer.train()
    metrics_df = pd.DataFrame(trainer.state.log_history)
    
    # Chama a função para salvar as métricas organizadas no CSV
    salvar_metricas_csv(metrics_df)

    # Gera o gráfico de época x acurácia
    plotar_grafico_acuracia_novo(caminho_csv="./runs/metrics_acumuladas.csv", caminho_imagem="./runs/epoch_vs_accuracy_acumulado.png")

if __name__ == "__main__":
    executar_fine_tuning()
