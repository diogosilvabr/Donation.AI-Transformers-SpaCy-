from flask import Flask
# Importa as rotas definidas em routes.py
from .routes import api_bp

# Registra o blueprint com o prefixo '/api'
def create_app():
    app = Flask(__name__)
    app.register_blueprint(api_bp, url_prefix='/api')  
    return app
