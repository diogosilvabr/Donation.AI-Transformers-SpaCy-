from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import spacy
import string
import re

# modelo spacy para PT-BR
nlp = spacy.load("pt_core_news_sm")

def preprocessarTexto(texto):
    # Remove pontuações
    texto = texto.translate(str.maketrans('', '', string.punctuation)) 
    # Converte para minúsculas
    texto = texto.lower() 
    # Remove números
    texto = re.sub(r'\d+', '', texto)  
    # Remove espaços em branco extras
    texto = texto.strip()
    # Processa o texto com SpaCy
    doc = nlp(texto) 
    # Lematiza e remove stop words
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    # Junta os tokens de volta em uma string
    texto_processado = " ".join(tokens)  
    return texto_processado

# Função para classificação de texto usando o modelo fine-tuned
def classificarTextoInadequado(texto):
    # Carregar o modelo e tokenizer fine-tuned
    modelo_caminho = "ml/modelo_finetuned"
    tokenizer = BertTokenizer.from_pretrained(modelo_caminho)
    modelo = BertForSequenceClassification.from_pretrained(modelo_caminho)

    classificador = pipeline("text-classification", model=modelo, tokenizer=tokenizer)
    resultado = classificador(texto)
    return resultado
