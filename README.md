# Análise de Sentimentos

## Descrição do Projeto
Este projeto utiliza técnicas de processamento de linguagem natural (NLP) para realizar análises de sentimento em textos. A aplicação identifica emoções predominantes (positivas, negativas ou neutras) e gera insights úteis para tomada de decisões baseadas em texto.

## Tecnologias Utilizadas
- **Python**: Linguagem principal.
- **NLTK**: Tokenização e pré-processamento de texto.
- **Pandas**: Manipulação de dados.
- **Matplotlib e Seaborn**: Visualização de dados.
- **Scikit-learn**: Modelo de aprendizado de máquina para classificação.

## Como Executar o Projeto
1. Clone o repositório:
   ```bash
   git clone https://github.com/seuusuario/analise-de-sentimento.git
   cd analise-de-sentimento
   ```
2. Instale os pacotes necessários:
   ```bash
   pip install -r requirements.txt
   ```
3. Inicie o script principal:
   ```bash
   python main.py
   ```

## Estrutura do Repositório
```
analise-de-sentimento/
├── data/                # Dados de entrada
├── notebooks/           # Notebooks exploratórios
├── src/                 # Código-fonte do projeto
│   ├── preprocess.py    # Funções de pré-processamento
│   ├── model.py         # Construção do modelo
│   └── utils.py         # Funções utilitárias
├── visuals/             # Gráficos gerados
├── README.md            # Documentação principal
└── requirements.txt     # Dependências do projeto
```

## Exemplos Visuais

### Distribuição de Sentimentos
![Distribuição de Sentimentos](/images/DistribuicaoTopicos.png))

### Palavra Nuvem
![Word Cloud](/images/durkheimWordcloud.png)

## Uso do Modelo
Exemplo de entrada e saída:

Entrada:
```python
from src.model import predict_sentiment

texto = "Adorei o filme, foi uma experiência incrível!"
predicao = predict_sentiment(texto)
print(predicao)
```

Saída:
```json
{"sentimento": "positivo", "confianca": 0.87}
```

## Contribuições
Contribuições são bem-vindas! Siga os passos abaixo para contribuir:
1. Faça um fork do repositório.
2. Crie uma nova branch:
   ```bash
   git checkout -b minha-feature
   ```
3. Commit suas mudanças:
   ```bash
   git commit -m "Descrição da minha feature"
   ```
4. Envie para a branch principal:
   ```bash
   git push origin minha-feature
   ```
5. Abra um Pull Request.

## Licença
Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais informações.


