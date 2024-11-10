from transformers import pipeline

# Carregar o modelo treinado
model = pipeline("text-classification", model="path_to_trained_model")  # Substitua pelo caminho correto do modelo

def preprocessarTexto(texto):
    # Função de pré-processamento (ajuste conforme necessário)
    return texto.lower()

def classificarTextoInadequado(texto_processado):
    # Fazer a classificação do texto usando o modelo treinado
    resultado = model(texto_processado)
    return resultado[0]['label']
