# Analise-de-sentimento


Análise de Sentimentos com NLP
Este projeto é uma análise de sentimentos baseada em Processamento de Linguagem Natural (NLP) de textos. A partir de técnicas de análise de sentimentos, modelagem de tópicos e visualizações, o objetivo é comparar diferentes textos, identificar padrões emocionais e explorar como modelos simples de NLP, como VADER e TextBlob, podem ser aplicados para entender nuances semânticas.

Funcionalidades
Leitura e Processamento de Texto: Leitura e extração de texto de arquivos PDF, com pré-processamento (remoção de pontuação, lematização, etc.).
Análise de Sentimentos: Uso dos modelos VADER e TextBlob para análise de sentimentos em textos curtos e longos.
Modelagem de Tópicos: Uso do LDA (Latent Dirichlet Allocation) para identificar tópicos nos textos e gerar uma visão mais detalhada sobre a estrutura semântica.
Visualização Interativa: Geração de nuvens de palavras, redes de palavras interativas e análise de sentimentos ao longo do tempo.
Análise de Similaridade entre Textos: Comparação de documentos usando vetores de tópicos e cálculo de similaridade de cosseno.
Detecção de Comunidades: Aplicação do algoritmo Louvain para detectar comunidades em redes de palavras.
Como Usar
Clone o repositório:
bash
Copiar código
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
Instale as dependências:
bash
Copiar código
pip install -r requirements.txt
Suba seu arquivo PDF: Coloque o arquivo PDF que deseja analisar na pasta do projeto.

Rodando o Código:

Execute o script principal para começar a análise de sentimentos:

bash
Copiar código
python main.py
O código irá:

Processar o texto do arquivo PDF.
Realizar a análise de sentimentos.
Gerar as visualizações.
Resultado: As visualizações (nuvem de palavras, gráficos de tópicos e redes de palavras) serão salvas ou exibidas conforme configurado no script.
Dependências
Python 3.x
nltk para processamento de linguagem natural
pandas para manipulação de dados
matplotlib e seaborn para visualizações
plotly para visualizações interativas
sklearn para modelos de aprendizado de máquina (LDA, KMeans)
vaderSentiment e textblob para análise de sentimentos
Resultados
O projeto pode ser usado para:

Analisar sentimentos em textos e comparar a polaridade (positivo, negativo, neutro).
Explorar tópicos ocultos em textos utilizando LDA.
Visualizar e analisar redes de palavras e como elas se conectam nos textos.
Comparar a similaridade de dois ou mais documentos.
Insights e Desafios
Este projeto é uma exploração em andamento e ainda está sendo expandido. Alguns dos desafios encontrados durante o desenvolvimento incluem:

Tempo de Performance: O processamento de textos muito grandes (como livros) pode ser lento, sendo necessário otimizar o uso de memória.
Análise de Sentimentos: Os modelos usados (VADER e TextBlob) funcionam bem para textos curtos, mas podem ser imprecisos para textos mais filosóficos e complexos, como análises críticas.
Limitações do Modelo: O modelo de análise de sentimentos pode não capturar as nuances de sentimentos em textos críticos ou filosóficos. A intenção aqui é melhorar a precisão da análise com mais testes e ajustes.
Exemplos de Resultados
Texto A: Após a análise, o modelo indicou uma polaridade negativa relacionada a críticas sobre religião.
Texto B: A análise revelou um sentimento neutro em um texto de uma escritora sobre a banalidade do mal, que possui uma abordagem mais filosófica e política.
Contribuições
Se você tem sugestões de melhorias ou gostaria de contribuir, fique à vontade para abrir uma issue ou enviar um pull request.
