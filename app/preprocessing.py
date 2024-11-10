from transformers import pipeline

# Carrega o modelo de classificação de texto do Hugging Face Transformers
model = pipeline("text-classification", model="./runs/checkpoint-1200")

def preprocessarTexto(texto):
    # Converte o texto para minúsculas com o pré-processamento básico
    return texto.lower()

def classificarTextoInadequado(texto_processado):
    # Classifica o texto pré-processado usando o modelo
    resultado = model(texto_processado)
    label = resultado[0]['label']  # Obtém o rótulo de classificação ("LABEL_0 = Adequado" ou "LABEL_1" = Inadequeado) 
    return label
