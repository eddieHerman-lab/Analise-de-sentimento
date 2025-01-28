import PyPDF2
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
import networkx as nx
from collections import defaultdict, Counter
import community.community_louvain as community_louvain
from pyvis.network import Network
import plotly.graph_objs as go
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

# Função para ler e filtrar texto de PDFs
def process_pdf(file_path):
    from PyPDF2 import PdfReader

    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():  # Verifica se a página não está vazia
                        text += page_text + "\n"
                    else:
                        print(f"A página {i + 1} está vazia ou não pode ser lida.")
                except Exception as e:
                    print(f"Erro ao processar a página {i + 1}: {e}")
    except Exception as e:
        print(f"Erro ao abrir o arquivo PDF: {e}")

    return text
# Limpeza do texto
def clean_text(text, use_lemmatization=True):
    if not text or len(text.strip()) == 0:
        print("Texto vazio recebido para limpeza.")
        return []

    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\[\d+\]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()

    tokens = word_tokenize(text)

    stop_words_pt = set(stopwords.words('portuguese'))
    stop_words_en = set(stopwords.words('english'))
    stop_words = stop_words_pt.union(stop_words_en)

    words = [word for word in tokens if word not in stop_words and len(word) > 2]


    if not words:
        print('Nenhuma palavra restante apos a remoção de stopwords:')
        return []

    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(token) for token in words]

    return words

# Funções de Análise e Visualização
def word_frequency(words):
    df = pd.DataFrame(words, columns=['word'])
    freq = df.groupby('word').size().reset_index(name='frequency')
    return freq.sort_values('frequency', ascending=False)

def sentiment_analysis(words,vader=True):
    text = ' '.join(words)
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if vader:
        # Inicializando o vader
        vader_analizer= SentimentIntensityAnalyzer()
        # calcula as pontuacoes de sentimento
        scores= vader_analizer.polarity_scores(text)
        compound= scores['compound']
        #print(f'Resultado Compound {compound} scores: {scores}')

        # Interpretacao com base no compound:
        if compound >=0.05:
            return ' Positivo'
        elif compound <=-0.05:
            return 'Negativo'
        else:
            return 'Neutro'
    return 'Positivo' if sentiment > 0 else 'Negativo' if sentiment < 0 else 'Neutro'

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

# LDA e Coerência
def find_optimal_number_of_topics(clean_text, start=2, limit=5, step=1):
    dictionary = Dictionary([clean_text])

    corpus = [dictionary.doc2bow(clean_text)]

    coherence_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=15)
        model_list.append(model)
        coherence_model_lda = CoherenceModel(model=model, texts=[clean_text], dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model_lda.get_coherence())

    return model_list, coherence_values

def plot_coherence(coherence_values, start=5, limit=20, step=5):
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Número de Tópicos")
    plt.ylabel("Coerência")
    plt.title("Coerência em relação ao número de tópicos")
    plt.show()

def word_network_with_weights(text, lda_model, dictionary):
    text = ' '.join(text)
    words = text.split()
    word_pairs = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    weights = defaultdict(float)

    for word1, word2 in word_pairs:
        bow1 = dictionary.doc2bow([word1])
        bow2 = dictionary.doc2bow([word2])
        for topic_num, prob in lda_model.get_document_topics(bow1, minimum_probability=0):
            for topic_num2, prob2 in lda_model.get_document_topics(bow2, minimum_probability=0):
                if topic_num == topic_num2:
                    weights[(word1, word2)] += (prob + prob2) / 2

    G = nx.Graph()
    for (word1, word2), weight in weights.items():
        if weight > 0:
            G.add_edge(word1, word2, weight=weight)
    return G

def filter_nodes_by_degree_centrality(G, threshold=0.01):
    centrality = nx.degree_centrality(G)
    nodes_to_remove = [n for n, c in centrality.items() if c < threshold]
    G_filtered = G.copy()
    G_filtered.remove_nodes_from(nodes_to_remove)
    return G_filtered

def filter_edges_by_weight(G, weight_threshold=0.5):
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 0) < weight_threshold]
    G_filtered = G.copy()
    G_filtered.remove_edges_from(edges_to_remove)
    return G_filtered

def detect_comunnities_louvain(G):
    partition = community_louvain.best_partition(G)
    return partition

def filter_graph_by_community(G, communities, top_n=3):
    communities_counts = Counter(communities.values())
    top_communities = [c for c, _ in communities_counts.most_common(top_n)]
    nodes_to_keep = [n for n in G.nodes if communities[n] in top_communities]
    G_filtered = G.subgraph(nodes_to_keep)
    return G_filtered
# Comparando documentos
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compare_documents(doc1, doc2, lda_model1, lda_model2, dictionary1, dictionary2):
    # Gerar vetores de tópicos para o primeiro documento
    bow1 = dictionary1.doc2bow(doc1)
    vector1 = [prob for _, prob in lda_model1.get_document_topics(bow1, minimum_probability=0)]

    # Gerar vetores de tópicos para o segundo documento
    bow2 = dictionary2.doc2bow(doc2)
    vector2 = [prob for _, prob in lda_model2.get_document_topics(bow2, minimum_probability=0)]

    # Padronizar os vetores para o mesmo tamanho
    max_len = max(len(vector1), len(vector2))
    vector1 = np.pad(vector1, (0, max_len - len(vector1)))
    vector2 = np.pad(vector2, (0, max_len - len(vector2)))

    # Calcular a similaridade de cosseno
    similarity = cosine_similarity([vector1], [vector2])[0][0]
    return similarity


#Predicao de sentimentos ao longo do tempo no texto
# Criacaod e uma janela movel
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

def detect_communities(G, num_communities=3):
    from networkx.algorithms.community import girvan_newman
    communities_generator = girvan_newman(G)
    limited = []
    try:
        for communities in communities_generator:
            limited = communities
            if len(limited) >= num_communities:
                break
    except StopIteration:
        pass
    return [list(community) for community in limited]

def compute_degree_centrality(G):
    degree_centrality = nx.degree_centrality(G)
    return degree_centrality


def extract_keywords_tfidf(text, n_keywords=10):
    """
    Extrai palavras-chave usando TF-IDF.

    Args:
    text (str): Texto para extrair palavras-chave.
    n_keywords (int): Número de palavras-chave a retornar.

    Returns:
    list: Lista das principais palavras-chave.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    word_scores = list(zip(feature_names, tfidf_scores))
    sorted_word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_word_scores[:n_keywords]]


def cluster_sentences(text, n_clusters=3):
    """
        Agrupa sentenças similares usando K-means.

        Args:
        text (str): Texto completo.
        n_clusters (int): Número de clusters a criar.

        Returns:
        tuple: Lista de sentenças e seus respectivos rótulos de cluster.
        """
    sentences = sent_tokenize(text)

    # Verifique o número de sentenças antes de aplicar K-Means
    if len(sentences) < n_clusters:
        n_clusters = len(sentences)  # Ajusta o número de clusters ao número de sentenças

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)

    return sentences, kmeans.labels_


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


def process_document(file_path, document_name=""):
    """Processa um único documento e retorna seus dados processados"""
    text = process_pdf(file_path)
    if not text.strip():
        print(f"{document_name} PDF está vazio ou não foi possível extrair o texto.")
        return None

    print(f"Texto extraído com sucesso do {document_name} PDF!")
    clean_words = clean_text(text, use_lemmatization=True)

    if not clean_words:
        print(f"Nenhuma palavra restante após limpeza no {document_name}")
        return None

    return {
        'text': text,
        'clean_words': clean_words,
        'word_freq': word_frequency(clean_words),
        'sentiment': sentiment_analysis(clean_words, vader=True),
        'model_data': find_optimal_number_of_topics(clean_words),
        'dictionary': Dictionary([clean_words])
    }


def analyze_document(doc_data, doc_name=""):
    """Realiza análises em um único documento"""
    if not doc_data:
        return None

    print(f"\n=== Análise do {doc_name} ===")
    print(f"Frequência de palavras (Top 20):")
    print(doc_data['word_freq'].head(20))

    generate_wordcloud(doc_data['clean_words'])
    print(f"Sentimento geral: {doc_data['sentiment']}")

    model_list, coherence_values = doc_data['model_data']
    plot_coherence(coherence_values)

    optimal_model_index = coherence_values.index(max(coherence_values))
    lda_model = model_list[optimal_model_index]

    print("Tópicos principais:")
    for topic in lda_model.print_topics(num_words=5):
        print(topic)

    # Análise de rede
    G = word_network_with_weights(doc_data['clean_words'], lda_model, doc_data['dictionary'])
    print(f"Rede: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas")

    G_filtered = filter_edges_by_weight(
        filter_nodes_by_degree_centrality(G, threshold=0.01),
        weight_threshold=0.5
    )

    plot_word_network_interactive(G_filtered)
    plot_topic_distribution(lda_model, doc_data['dictionary'],doc_data['clean_words'])

    # Análise temporal e clustering
    sentiment_over_time(doc_data['clean_words'], window_size=True)
    keywords = extract_keywords_tfidf(' '.join(doc_data['clean_words']))
    print("Palavras-chave extraídas:", keywords)

    communities = detect_communities(G_filtered, num_communities=5)
    print("Comunidades detectadas:")
    for idx, community in enumerate(G_filtered, 1):
        print(f"Comunidade {idx}: {community}")

    sentences, labels = cluster_sentences(doc_data['text'], n_clusters=3)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    plot_sentence_clusters(sentences, labels, X)

    return {
        'lda_model': lda_model,
        'dictionary': doc_data['dictionary'],
        'clean_words': doc_data['clean_words']
    }


def compare_documents_analysis(doc1_data, doc2_data):
    """Realiza análises comparativas entre dois documentos"""
    if not (doc1_data and doc2_data):
        print("Impossível comparar documentos - dados ausentes")
        return

    similarity = compare_documents(
        doc1_data['clean_words'],
        doc2_data['clean_words'],
        doc1_data['lda_model'],
        doc2_data['lda_model'],
        doc1_data['dictionary'],
        doc2_data['dictionary']
    )
    print(f"\nSimilaridade entre os documentos: {similarity:.4f}")


def main():
    file_paths = {
        "Documento 1": "C:/Users/55219/Desktop/DURKHEIM E AS FORMAS ELEMENTARES.Limpo.pdf",
        "Documento 2": "C:/Users/55219/Desktop/Tratado_da_Natureza_Humana_David_Hume.pdf"
    }

    # Processamento inicial
    processed_docs = {
        name: process_document(path, name)
        for name, path in file_paths.items()
    }

    # Análise individual
    analyzed_docs = {
        name: analyze_document(doc_data, name)
        for name, doc_data in processed_docs.items()
    }

    # Análise comparativa
    if all(analyzed_docs.values()):
        compare_documents_analysis(
            analyzed_docs["Documento 1"],
            analyzed_docs["Documento 2"]
        )


if __name__ == "__main__":
    main()
