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

