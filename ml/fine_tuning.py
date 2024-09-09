from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd
from transformers import EarlyStoppingCallback
import logging

# Configuração de logs
logging.basicConfig(level=logging.INFO)

# Função para carregar o modelo e o tokenizer
def carregar_modelo_tokenizer(modelo_caminho="neuralmind/bert-base-portuguese-cased"):
    logging.info(f"Carregando o modelo e o tokenizer de {modelo_caminho}")
    tokenizer = BertTokenizer.from_pretrained(modelo_caminho, do_lower_case=False)
    modelo = BertForSequenceClassification.from_pretrained(modelo_caminho, num_labels=2)
    return modelo, tokenizer

# Função para carregar e processar o dataset
def carregar_dataset(caminho_csv="../data/base.csv"):
    logging.info(f"Carregando dataset de {caminho_csv}")
    dataset = load_dataset("csv", data_files={"train": caminho_csv})
    return dataset

# Função para tokenizar o dataset
def tokenizar_dataset(dataset, tokenizer):
    logging.info("Tokenizando o dataset")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("inappropriate", "labels")  # Renomear coluna de rótulos
    return tokenized_datasets

# Função para configurar o treinamento
def configurar_treinamento(modelo, tokenized_datasets, output_dir="./results"):
    logging.info("Configurando o treinamento")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True
    )
    
    # Definindo o Trainer com Early Stopping
    trainer = Trainer(
        model=modelo,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    return trainer

# Função principal para rodar o fine-tuning
def executar_fine_tuning():
    # Carregar o modelo e tokenizer
    modelo, tokenizer = carregar_modelo_tokenizer()

    # Carregar e tokenizar o dataset
    dataset = carregar_dataset()
    tokenized_datasets = tokenizar_dataset(dataset, tokenizer)

    # Configurar e iniciar o treinamento
    trainer = configurar_treinamento(modelo, tokenized_datasets)
    
    logging.info("Iniciando o treinamento")
    trainer.train()

    # Salvar o modelo e tokenizer treinados
    modelo.save_pretrained("modelo_finetuned")
    tokenizer.save_pretrained("modelo_finetuned")
    logging.info("Modelo e tokenizer salvos com sucesso em 'modelo_finetuned'.")

if __name__ == "__main__":
    executar_fine_tuning()
