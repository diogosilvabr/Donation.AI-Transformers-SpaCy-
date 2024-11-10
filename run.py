# Importa CORS para permitir requisições de origens diferentes
from flask_cors import CORS  
# Importa a função de criação do app
from app import create_app 

# Cria a aplicação Flask e adiciona suporte a CORS
app = create_app()
CORS(app)

# Executa a aplicação no modo debug na porta 8888
if __name__ == "__main__":

    app.run(debug=True, port=8888)
