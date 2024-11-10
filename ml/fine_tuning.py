import logging
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
import evaluate
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback

# Função para calcular a acurácia
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return {"accuracy": accuracy["accuracy"]}

# Função para coletar e armazenar métricas de treino e avaliação
class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_metrics = []
        self.eval_metrics = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'eval_accuracy' in logs:
                self.eval_metrics.append({'epoch': state.epoch, 'accuracy': logs['eval_accuracy']})
            if 'accuracy' in logs:
                self.train_metrics.append({'epoch': state.epoch, 'accuracy': logs['accuracy']})

metrics_callback = MetricsCallback()

# Função para salvar as métricas em um arquivo CSV
def salvar_metricas_csv(metrics_callback, arquivo_csv="./runs/metrics_acumuladas.csv"):
    # Converte os dados de treino e validação para DataFrame
    df_train = pd.DataFrame(metrics_callback.train_metrics)
    df_train['tipo'] = 'treino'
    df_eval = pd.DataFrame(metrics_callback.eval_metrics)
    df_eval['tipo'] = 'validação'
    
    # Concatena os dados de treino e validação
    df = pd.concat([df_train, df_eval])
    
    # Salva ou adiciona ao arquivo CSV
    if os.path.exists(arquivo_csv):
        df_antigo = pd.read_csv(arquivo_csv)
        df = pd.concat([df_antigo, df])
    
    df.to_csv(arquivo_csv, index=False)

# Função para carregar métricas anteriores de um CSV
def carregar_metricas_csv(arquivo_csv="metrics_acumuladas.csv"):
    if os.path.exists(arquivo_csv):
        return pd.read_csv(arquivo_csv)
    return pd.DataFrame()

# Função para plotar o gráfico de época vs acurácia acumulado
def plot_epoch_vs_accuracy_acumulado(arquivo_csv="./runs/metrics_acumuladas.csv"):
    df = carregar_metricas_csv(arquivo_csv)
    
    if df.empty:
        print("Sem métricas para exibir.")
        return
    
    # Plotar as métricas acumuladas
    for tipo in df['tipo'].unique():
        df_tipo = df[df['tipo'] == tipo]
        plt.plot(df_tipo['epoch'], df_tipo['accuracy'], label=f'{tipo.capitalize()} - Acurácia', linestyle='--' if tipo == 'validação' else '-')
    
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.title('Épocas x Acurácia - Treinamentos Acumulados')
    plt.legend()
    plt.grid(True)
    plt.savefig('./runs/epoch_vs_accuracy_acumulado.png')
    plt.show()

# Função para carregar o dataset unificado
def carregar_dataset(caminho_csv="../data/base_unificada.csv"):
    caminho_absoluto = os.path.join(os.path.dirname(os.path.abspath(__file__)), caminho_csv)
    
    logging.info(f"Carregando dataset de {caminho_absoluto}")
    
    dataset = load_dataset("csv", data_files=caminho_absoluto)
    
    # Renomeia a coluna de rótulos (garantindo que a coluna 'inappropriate' seja tratada como 'labels')
    dataset = dataset['train'].rename_column("inappropriate", "labels")
    
    # Divide em treino e validação
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    return dataset

# Função para tokenizar o dataset
def tokenizar_dataset(dataset, tokenizer):
    logging.info("Tokenizando o dataset")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="longest", truncation=True, max_length=60)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets

# Função para configurar o treinamento
def configurar_treinamento(modelo, tokenized_datasets, tokenizer):
    logging.info("Configurando o treinamento")

    # Definir os argumentos de treinamento
    training_args = TrainingArguments(
        output_dir="./runs",
        logging_dir="./runs",
        evaluation_strategy="steps",
        save_strategy="steps", 
        per_device_train_batch_size=8,  
        per_device_eval_batch_size=8,
        num_train_epochs=10,  
        weight_decay=0.01,
        save_steps=200,
        eval_steps=200,
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=100,  
        learning_rate=5e-5,  
        lr_scheduler_type="linear",  
        warmup_steps=500,  
        gradient_accumulation_steps=1,  
    )

    # Configurar o Trainer com early stopping e um scheduler
    trainer = Trainer(
        model=modelo,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), metrics_callback],  # Adiciona o metrics_callback
    )
    
    return trainer

# Função principal para executar o fine-tuning
def executar_fine_tuning():
    logging.info("Carregando o modelo e o tokenizer de neuralmind/bert-base-portuguese-cased")
    
    # Verificar se a GPU está disponível e usá-la
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    modelo = BertForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased", num_labels=2)
    
    # Enviar o modelo para a GPU, se disponível
    modelo.to(device)
    
    dataset = carregar_dataset()
    tokenized_datasets = tokenizar_dataset(dataset, tokenizer)
    
    trainer = configurar_treinamento(modelo, tokenized_datasets, tokenizer)
    
    logging.info("Iniciando o treinamento")
    trainer.train()

    # Salvar as métricas em um arquivo CSV
    salvar_metricas_csv(metrics_callback)
    
    # Chamar a função para plotar o gráfico acumulado após o treinamento
    plot_epoch_vs_accuracy_acumulado()


if __name__ == "__main__":
    executar_fine_tuning()
