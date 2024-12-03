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

# Ler o arquivo PDF
def read_pdf(file_path):
    with open(file_path , 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Limpeza do texto
def clean_text(text, use_lemmatization=True):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\[\d+\]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words_pt = set(stopwords.words('portuguese'))
    stop_words_en = set(stopwords.words('english'))
    stop_words = stop_words_pt.union(stop_words_en)
    words = [word for word in tokens if word not in stop_words and len(word) > 2]

    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(token) for token in words]

    return words
