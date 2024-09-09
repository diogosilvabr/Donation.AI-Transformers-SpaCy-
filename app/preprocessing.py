import spacy
import string
import re
from transformers import pipeline, BertTokenizer, BertForSequenceClassification

# Carregar o SpaCy uma vez
nlp = spacy.load("pt_core_news_sm")

# Carregar o modelo e tokenizer uma vez
modelo_caminho = "ml/modelo_finetuned"
tokenizer = BertTokenizer.from_pretrained(modelo_caminho)
modelo = BertForSequenceClassification.from_pretrained(modelo_caminho)

# Função de pré-processamento do texto
def preprocessarTexto(texto):
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto).strip()
    
    # Processamento com SpaCy
    doc = nlp(texto)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

# Função para classificar o texto usando o modelo fine-tuned
def classificarTextoInadequado(texto):
    classificador = pipeline("text-classification", model=modelo, tokenizer=tokenizer)
    resultado = classificador(texto)
    return resultado
