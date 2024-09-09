from flask_cors import CORS
from app import create_app

# Cria a aplicação Flask
app = create_app()

# Adiciona suporte a CORS
CORS(app)

if __name__ == "__main__":
    # Executa a aplicação no modo debug na porta 8888
    app.run(debug=True, port=8888)
