import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import (
    BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, 
    EarlyStoppingCallback
)
from datasets import load_dataset

# Função para calcular a acurácia como métrica
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    accuracy = (predictions == labels).astype(float).mean().item()
    return {"accuracy": accuracy}

# Função para carregar e tokenizar o dataset
def preparar_dados(caminho_csv="../data/base_unificada.csv"):
    dataset = load_dataset("csv", data_files=caminho_csv)['train']
    dataset = dataset.rename_column("inappropriate", "labels").train_test_split(test_size=0.2, seed=42)
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    return dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=60), batched=True)

# Função para plotar o gráfico de época x acurácia e salvar como imagem
def plotar_grafico_acuracia(df_metrics, caminho="./runs/epoch_vs_accuracy_acumulado.png"):
    df_metrics.plot(x="epoch", y="accuracy", kind="line", title="Épocas x Acurácia")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.savefig(caminho)
    plt.show()

# Configura o treinamento, executa o fine-tuning e salva as métricas
def executar_fine_tuning():
    modelo = BertForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased", num_labels=2)
    dataset = preparar_dados()
    
    trainer = Trainer(
        model=modelo,
        args=TrainingArguments(
            output_dir="./runs",  # Diretório onde os modelos serão salvos
            evaluation_strategy="epoch",  # Avaliação a cada época
            save_strategy="epoch",  # Salvamento do modelo a cada época
            logging_dir="./logs",  # Diretório dos logs detalhados
            num_train_epochs=15,  # Número total de épocas
            per_device_train_batch_size=8,  # Tamanho do batch de treinamento
            per_device_eval_batch_size=8,  # Tamanho do batch de avaliação
            weight_decay=0.01,  # Fator de decaimento do peso
            load_best_model_at_end=True,  # Carregar o melhor modelo ao final
            logging_steps=10,  # Frequência de log detalhado para monitoramento
            save_total_limit=2  # Limite de modelos salvos para evitar excesso de arquivos
        ),
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Para interromper o treinamento se não houver melhoria
    )

    # Executa o treinamento e armazena as métricas em CSV
    train_result = trainer.train()
    metrics_df = pd.DataFrame(trainer.state.log_history)
    metrics_df.to_csv("./runs/metrics_acumuladas.csv", index=False)  # Salva métricas detalhadas em CSV

    # Gera o gráfico de época x acurácia
    plotar_grafico_acuracia(metrics_df)

if __name__ == "__main__":
    executar_fine_tuning()
