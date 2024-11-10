
# Donation.IA

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

Este projeto utiliza modelos de machine learning, especificamente Transformers com BERT e SpaCy, para analisar textos e identificar linguagem inadequada.

## Índice

- [Visão Geral](#visão-geral)
- [Instalação](#instalação)
- [Uso](#uso)
- [Fine-Tuning](#fine-tuning)
- [Contribuição](#contribuição)
- [Licença](#licença)

## Visão Geral

O objetivo deste projeto é criar uma API que analise textos para identificar se contêm linguagem inadequada. O modelo baseia-se no BERT pré-treinado para o português e é fine-tuned com um conjunto de dados específico para este projeto. O SpaCy também é utilizado para pré-processamento de texto e análise de entidades.

## Instalação

Siga estas etapas para configurar e executar o projeto localmente.

### Pré-requisitos

- Python 3.8 ou superior
- Pip (gerenciador de pacotes do Python)

### Passos

1. Clone o repositório:
   ```bash
   git clone https://github.com/diogosilvabr/Donation.AI-Transformers-SpaCy-.git
   cd Donation.AI-Transformers-SpaCy-
   ```

2. Crie um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate  # No Windows
   # Ou, no Linux/macOS
   source venv/bin/activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Execute a aplicação:
   ```bash
   python run.py
   ```

   O arquivo `run.py` já está configurado para rodar o aplicativo Flask em modo de depuração (`debug=True`), permitindo uma visualização detalhada dos erros e recarregamento automático durante o desenvolvimento.

## Uso

### Endpoints

#### Analisar Texto

- **URL**: `/analyze-text`
- **Método**: `POST`
- **Parâmetros de Entrada**: JSON com o campo `text`
- **Exemplo de Entrada**:
  ```json
  {
    "text": "Seu texto aqui"
  }
  ```
- **Exemplo de Saída**:
  ```json
  {
    "inapropriado": true
  }
  ```

## Fine-Tuning

O modelo BERT é fine-tuned utilizando um conjunto de dados específico armazenado em `data/base.csv`. O treinamento salva logs e métricas automaticamente, e gera um gráfico de acurácia por época.

Para realizar o fine-tuning:

1. Certifique-se de que o dataset está no formato correto.
2. Execute o script de fine-tuning:
   ```bash
   python ml/fine_tuning.py
   ```

   - O modelo ajustado será salvo no diretório `runs/`.
   - As métricas serão salvas em `runs/metrics_acumuladas.csv`.
   - Um gráfico de acurácia por época será gerado automaticamente como `runs/epoch_vs_accuracy_acumulado.png`.

## Contribuição

Contribuições são bem-vindas! Se você tiver sugestões, encontrar bugs ou quiser contribuir com o código, por favor, siga estes passos:

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/sua-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/sua-feature`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

## Autor

[Diogo Silva](https://github.com/diogosilvabr)

## Referências

- A base de dados [ToLD-BR](https://github.com/JAugusto97/ToLD-Br) foi utilizada neste projeto para melhorar a precisão do modelo de machine learning.
