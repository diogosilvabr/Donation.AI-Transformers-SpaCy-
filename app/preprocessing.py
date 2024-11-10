from transformers import pipeline

# Carregar o modelo treinado
model = pipeline("text-classification", model="./runs/checkpoint-1200")  # Substitua pelo caminho correto do modelo

def preprocessarTexto(texto):
    # Função de pré-processamento (ajuste conforme necessário)
    return texto.lower()

def classificarTextoInadequado(texto_processado):
    # Fazer a classificação do texto usando o modelo treinado
    resultado = model(texto_processado)
    
    # Extraia o rótulo de `resultado`, assumindo que ele retorne uma lista de dicionários
    label = resultado[0]['label']  # Isso deve retornar "label_0" ou "label_1"
    
    return label

