import logging
from flask import Blueprint, request, jsonify
from .preprocessing import preprocessarTexto, classificarTextoInadequado

# Configuração de logs
logging.basicConfig(level=logging.INFO)

# Cria o blueprint da API
api_bp = Blueprint('api', __name__)

# Mapeamento dos rótulos para termos de melhor entendimento
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

    # Pré-processa e classifica o texto
    texto_processado = preprocessarTexto(texto)
    classificacao = classificarTextoInadequado(texto_processado)
    logging.info(f"Classificação bruta: {classificacao}")

    # Mapeia o rótulo para uma classificação legível
    classificacao_legivel = label_map.get(classificacao, classificacao)
    logging.info(f"Classificação legível: {classificacao_legivel}")

    return jsonify({
        'texto_processado': texto_processado,
        'classificacao': classificacao_legivel
    })
