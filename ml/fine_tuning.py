from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoModelForPreTraining, AutoTokenizer
from datasets import load_dataset
import pandas as pd

# Carregar o tokenizer e o modelo para fine-tuning em classificação
tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = BertForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased", num_labels=2)

# Carregar a base de dados CSV em um DataFrame
df = pd.read_csv("../data/base.csv")

# Converter para o formato do dataset Hugging Face
dataset = load_dataset("csv", data_files={"train": "../data/base.csv"})

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Certifique-se de que o conjunto de dados tenha as colunas corretas
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("inappropriate", "labels")


# Definir os argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Definir o objeto Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
)

# Iniciar o treinamento
trainer.train()

# Salvar o modelo fine-tuned
model.save_pretrained("modelo_finetuned")
tokenizer.save_pretrained("modelo_finetuned")
