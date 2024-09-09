from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Carregar a base de dados CSV em um DataFrame
df = pd.read_csv("data/base.csv")

# Dividir os dados em treino e teste
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Salvar os conjuntos de dados separados
train_df.to_csv("ml/train.csv", index=False)
test_df.to_csv("ml/test.csv", index=False)

# Carregar o tokenizer e o modelo para fine-tuning em classificação
tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = BertForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased", num_labels=2)

# Carregar a base de dados CSV em DataFrames e convertê-los para o formato do Hugging Face
dataset = load_dataset("csv", data_files={"train": "ml/train.csv", "test": "ml/test.csv"})

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Certifique-se de que o conjunto de dados tenha as colunas corretas
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("inappropriate", "labels")

# Definir os argumentos de treinamento
training_args = TrainingArguments(
    output_dir="/results",
    eval_strategy="epoch",  # Correção aqui
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",  # Salvamento do modelo a cada época
    logging_dir="./logs",  # Diretório para logs
    logging_strategy="epoch",  # Log a cada época
    load_best_model_at_end=True,  # Carregar o melhor modelo no final do treinamento
)

# Definir o objeto Trainer com EarlyStoppingCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Adicionar EarlyStoppingCallback
)

# Iniciar o treinamento
trainer.train()

# Salvar o modelo fine-tuned
model.save_pretrained("modelo_finetuned")
tokenizer.save_pretrained("modelo_finetuned")
