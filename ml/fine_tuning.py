from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)

# Função para carregar o dataset e renomear a coluna de rótulos
def carregar_dataset(caminho_csv="data/base.csv"):
    logging.info(f"Carregando dataset de {caminho_csv}")
    dataset = load_dataset("csv", data_files=caminho_csv)
    
    # Renomear a coluna de rótulos para 'labels'
    dataset = dataset['train'].rename_column("inappropriate", "labels")
    
    return dataset

# Função para tokenizar o dataset
def tokenizar_dataset(dataset, tokenizer):
    logging.info("Tokenizando o dataset")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets


# Função para configurar o treinamento
def configurar_treinamento(modelo, tokenized_datasets, tokenizer):
    logging.info("Configurando o treinamento")
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",  # Mudar para 'steps'
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=500,  # Defina quando deseja salvar
        eval_steps=500,  # Defina quando deseja avaliar
        save_total_limit=1,
        load_best_model_at_end=True,  # Isso agora funcionará corretamente
)
    
    trainer = Trainer(
        model=modelo,
        args=training_args,
        train_dataset=tokenized_datasets,  # Remova ["train"] se não houver divisão
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
