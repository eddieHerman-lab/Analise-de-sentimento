import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import plotly.graph_objs as go
import seaborn as sns
import networkx as nx

# Nuvem de palavras
import logging
logging.basicConfig(level=logging.INFO)

def generate_wordcloud(words):
    text = ' '.join(words)
    if len(text.split()) == 0:
        print("O texto está vazio após a limpeza. A nuvem de palavras não pode ser gerada.")
        return
    wordcloud = WordCloud(width=800, height=600, background_color='white').generate(text)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('wordcloud.png')
    plt.show()
    plt.close()

def plot_coherence(coherence_values, start=5, limit=20, step=5):
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Número de Tópicos")
    plt.ylabel("Coerência")
    plt.title("Coerência em relação ao número de tópicos")
    plt.show()

#Predicao de sentimentos ao longo do tempo no texto
# Criacao de uma janela movel
def sentiment_over_time(text,window_size=100,window_topic=False):
    chunks= [text[i:i+window_size]for i in range(0,len(text))]
    sentiments= [sentiment_analysis(chunk) for chunk in chunks]

    #Transformar positivo, neutro e negativo em valores numericos
    sentiment_values=[1 if s=='Positivo' else -1 if s== 'Negativo' else 0 for s in sentiments]
    #Plotagem
    plt.plot(range(len(sentiments)), sentiment_values, marker='o')
    plt.title('Sentimento ao longo do texto')
    plt.xlabel('Partes do Texto')
    plt.ylabel('Sentimento (Positivo = 1, Neutro = 0, Negativo = -1)')
    plt.show()
    if window_topic == True:
        _,model_topics= find_optimal_number_of_topics(text)
        lenght_topic= len(model_topics)
        chunks = [text[i:i + lenght_topic] for i in range(0, len(text))]
        sentiments = [sentiment_analysis(chunk) for chunk in chunks]

        # Transformar positivo, neutro e negativo em valores numericos
        sentiment_values = [1 if s == 'Positivo' else -1 if s == 'Negativo' else 0 for s in sentiments]
        # Plotagem
        plt.plot(range(len(sentiments)), sentiment_values, marker='o')
        plt.title('Sentimento ao longo do texto')
        plt.xlabel('Partes do Texto')
        plt.ylabel('Sentimento (Positivo = 1, Neutro = 0, Negativo = -1)')
        plt.show()


def plot_word_network_interactive(G):
    pos = nx.spring_layout(G)
    edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers+text', textposition='top center', hoverinfo='text',
        marker=dict(showscale=True, colorscale='YlGnBu', size=10, colorbar=dict(thickness=15, title='Degree Centrality', xanchor='left', titleside='right'), line_width=2)
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    node_trace['text'] = list(G.nodes())
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title='<br>Network graph made with Python', titlefont_size=16, showlegend=False, hovermode='closest', margin=dict(b=0, l=0, r=0, t=40), xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False)))
    fig.show()
def plot_word_network_interactive_with_size(G):
    pos = nx.spring_layout(G)
    centrality = nx.degree_centrality(G)
    max_weight = max([d.get('weight', 1) for u, v, d in G.edges(data=True)])

    # Traçando as arestas
    edge_trace = go.Scatter(
        x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Ajustando as arestas de acordo com o peso
    edge_width = [G[u][v].get('weight', 1) / max_weight * 5 for u, v in G.edges()]

    # Traçando os nós
    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers+text', textposition='top center', hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=[v * 1000 for v in centrality.values()],  # Ajustando o tamanho dos nós com base na centralidade
            colorbar=dict(
                thickness=15, title='Degree Centrality', xanchor='left', titleside='right'
            ),
            line_width=2)
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    node_trace['text'] = list(G.nodes())

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title='<br>Network graph with adjusted node sizes and edge weights', titlefont_size=16, showlegend=False,
        hovermode='closest', margin=dict(b=0, l=0, r=0, t=40), xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)))

    fig.show()

def plot_sentence_clusters(sentences, labels,X):
    # Reduzir a dimensionalidade para 2D
    pca = PCA(n_components=2)
    reduced_X = pca.fit_transform(X.toarray())  # Convertendo matriz esparsa para densa

    # Criar DataFrame para plotagem
    df = pd.DataFrame({'x': reduced_X[:, 0], 'y': reduced_X[:, 1], 'cluster': labels})
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='viridis', s=100)
    plt.title('Componente PCA 1')
    plt.xlabel('Componente PCA 2')
    plt.ylabel('Cluster')
    plt.show()
    """
  Plota os clusters de sentenças.

   Args:
   sentences (list): Lista de sentenças.
   labels (list): Rótulos de cluster para cada sentença.
   """


def plot_topic_distribution(lda_model, dictionary, text):
    """
    Plota a distribuição de tópicos nos documentos.

    Args:
    lda_model: Modelo LDA treinado
    dictionary: Dicionário gensim
    text: Texto limpo (lista de palavras)
    """
    # Criar corpus para o texto
    bow = dictionary.doc2bow(text)

    # Inicializa a lista para armazenar os pesos de cada tópico
    topic_weights = []
    topic_dist = lda_model.get_document_topics(bow)
    weights = [0] * lda_model.num_topics
    for topic_num, weight in topic_dist:
        weights[topic_num] = weight
    topic_weights.append(weights)

    # Converte os pesos para um DataFrame
    df = pd.DataFrame(topic_weights).transpose()
    df.columns = ['Documento']

    # Extrai os principais termos de cada tópico
    topic_names = []
    for i in range(lda_model.num_topics):
        terms = lda_model.show_topic(i, topn=3)
        terms_str = ", ".join([term for term, _ in terms])
        topic_names.append(f"Tópico {i}: {terms_str}")

    df.index = topic_names

    # Plotagem
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, cmap='YlOrRd', cbar_kws={'label': 'Peso do Tópico'})
    plt.title('Distribuição de Tópicos no Documento', fontsize=16)
    plt.ylabel('Tópicos', fontsize=12)
    plt.xlabel('Documento', fontsize=12)
    plt.tight_layout()
    plt.show()
