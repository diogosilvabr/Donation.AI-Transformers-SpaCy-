import csv
import pandas as pd 
from flask import Blueprint, request, jsonify
from .preprocessing import preprocessarTexto, classificarTextoInadequado


# Cria um blueprint para organizar as rotas da API
api_bp = Blueprint('api', __name__)

# Cria a rota para o endpoint de análise de texto
@api_bp.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.json
    texto = data.get('text', '')

    # Recebe apenas o texto processado
    texto_processado = preprocessarTexto(texto)
    classificacao = classificarTextoInadequado(texto_processado)

    return jsonify({
        'texto_processado': texto_processado,
        'classificacao': classificacao
    })
