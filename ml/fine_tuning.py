from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
import logging
import torch
import os

logging.basicConfig(level=logging.INFO)

# Função para carregar o dataset e renomear a coluna de rótulos
def carregar_dataset(caminho_csv="../data/base.csv"):
    # Resolve o caminho absoluto
    caminho_absoluto = os.path.join(os.path.dirname(os.path.abspath(__file__)), caminho_csv)
    
    logging.info(f"Carregando dataset de {caminho_absoluto}")
    
    dataset = load_dataset("csv", data_files=caminho_absoluto)
    
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

    # Definir os argumentos de treinamento
    training_args = TrainingArguments(
        output_dir="./results",
        logging_dir="./runs",
        evaluation_strategy="steps",
        save_strategy="steps", 
        per_device_train_batch_size=8,  # Mantendo o batch size baixo para VRAM limitada
        per_device_eval_batch_size=8,
        num_train_epochs=5,  # Teste com 5 épocas inicialmente
        weight_decay=0.01,
        save_steps=200,
        eval_steps=200,
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=100,  # Log após cada 100 steps
        learning_rate=5e-5,  # Taxa de aprendizado inicial
        lr_scheduler_type="linear",  # Redução linear da taxa de aprendizado
        warmup_steps=500,  # Número de steps de aquecimento
        gradient_accumulation_steps=1,  # Acumula gradientes para reduzir o uso de VRAM
    )

    # Configurar o Trainer com early stopping e um scheduler
    trainer = Trainer(
        model=modelo,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Early stopping após 3 iterações sem melhoria
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


if __name__ == "__main__":
    executar_fine_tuning()
