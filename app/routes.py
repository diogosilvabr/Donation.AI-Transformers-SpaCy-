import logging
from flask import Blueprint, request, jsonify
from .preprocessing import preprocessarTexto, classificarTextoInadequado

# Configuração básica de logs
logging.basicConfig(level=logging.INFO)

# Blueprint da API
api_bp = Blueprint('api', __name__)

# Dicionário de mapeamento para tornar os rótulos mais legíveis
label_map = {
    "LABEL_0": "Adequado",
    "LABEL_1": "Inadequado"
}

@api_bp.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.json
    texto = data.get('text', '')
    
    if not texto:
        logging.error("Texto não fornecido")
        return jsonify({"erro": "Texto não fornecido"}), 400

    logging.info(f"Texto recebido: {texto}")

    # Pré-processar o texto
    texto_processado = preprocessarTexto(texto)
    
    # Classificar o texto usando o modelo treinado
    classificacao = classificarTextoInadequado(texto_processado)
    # Classificar o texto usando o modelo treinado
    logging.info(f"Classificação bruta retornada pelo modelo: {classificacao}")

    # Aplicar o mapeamento para tornar a classificação mais legível
    classificacao_legivel = label_map.get(classificacao, classificacao)

    logging.info(f"Classificação: {classificacao_legivel}")

    return jsonify({
        'texto_processado': texto_processado,
        'classificacao': classificacao_legivel
    })
