import logging
from flask import Blueprint, request, jsonify
from .preprocessing import preprocessarTexto, classificarTextoInadequado

# Configuração básica de logs
logging.basicConfig(level=logging.INFO)

api_bp = Blueprint('api', __name__)

@api_bp.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.json
    texto = data.get('text', '')

    # Log para verificar o texto recebido
    logging.info(f"Texto recebido: {texto}")

    # Pré-processar e classificar o texto
    texto_processado = preprocessarTexto(texto)
    classificacao = classificarTextoInadequado(texto_processado)

    # Log para verificar o resultado da classificação
    logging.info(f"Classificação: {classificacao}")

    return jsonify({
        'texto_processado': texto_processado,
        'classificacao': classificacao
    })
