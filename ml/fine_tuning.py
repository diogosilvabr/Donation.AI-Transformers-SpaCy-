from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import logging
import torch

logging.basicConfig(level=logging.INFO)

# Função para carregar o dataset e renomear a coluna de rótulos
def carregar_dataset(caminho_csv="../data/base.csv"):
    logging.info(f"Carregando dataset de {caminho_csv}")
    dataset = load_dataset("csv", data_files=caminho_csv)
    # Renomeia a coluna de rótulos
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
    
    training_args = TrainingArguments(
        output_dir="./results",
        logging_dir="./runs",
        evaluation_strategy="steps",  # Avalia a cada step definido em eval_steps
        save_strategy="steps",  # Salva a cada step definido em save_steps
        per_device_train_batch_size=4,  # Ajuste se houver problemas de memória na GPU
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        save_steps=200,
        eval_steps=200,
        save_total_limit=1,
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=modelo,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
    )
    
    return trainer

# Função principal para executar o fine-tuning
def executar_fine_tuning():
    logging.info("Carregando o modelo e o tokenizer de neuralmind/bert-base-portuguese-cased")
    
    tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    modelo = BertForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased", num_labels=2)
    
    dataset = carregar_dataset()
    tokenized_datasets = tokenizar_dataset(dataset, tokenizer)
    
    trainer = configurar_treinamento(modelo, tokenized_datasets, tokenizer)
    
    logging.info("Iniciando o treinamento")
    trainer.train()

if __name__ == "__main__":
    executar_fine_tuning()
