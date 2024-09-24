import spacy
import string
import re
from transformers import pipeline, BertTokenizer, BertForSequenceClassification

# Carregar o SpaCy e o modelo
nlp = spacy.load("pt_core_news_sm")
modelo_caminho = "ml/modelo_finetuned"
tokenizer = BertTokenizer.from_pretrained(modelo_caminho)
modelo = BertForSequenceClassification.from_pretrained(modelo_caminho)
classificador = pipeline("text-classification", model=modelo, tokenizer=tokenizer)

# Função de pré-processamento do texto
def preprocessarTexto(texto):
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto).strip()
    
    # Processamento com SpaCy
    doc = nlp(texto)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

# Função para classificar o texto usando o modelo fine-tuned
def classificarTextoInadequado(texto):
    resultado = classificador(texto)
    return resultado
