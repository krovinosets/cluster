import os
import psycopg2
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
from kneed import KneeLocator
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from src.cache import save_to_cache, is_cache_not_empty, load_from_cache
import re
import nltk
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer

nltk.download('stopwords')


def get_vk_posts_texts() -> tuple[list[str], list[int]]:
    conn = psycopg2.connect(
        **{
            'dbname': os.getenv('DATABASE_NAME'),
            'user': os.getenv('DATABASE_USER'),
            'password': os.getenv('DATABASE_PASSWORD'),
            'host': os.getenv('DATABASE_HOST'),
            'port': os.getenv('DATABASE_PORT')
        }
    )
    texts = []
    group_ids = []
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT text, group_id FROM vk_posts;")
            results = cursor.fetchall()
            texts = [row[0] for row in results]
            group_ids = [row[-1] for row in results]
    except psycopg2.Error as e:
        print(f"Ошибка при выполнении запроса: {e}")

    return texts, group_ids


def preprocess_texts(texts: list[str]) -> list[str]:
    morph = MorphAnalyzer()
    russian_stopwords = set(stopwords.words('russian') + [
        'этот', 'это', 'весь', 'который', 'такой', 'какой', 'наш', 'свой'
    ])

    def clean_text(text: str) -> str:
        text = re.sub(r'<[^>]+>', '', text)  # HTML-теги
        text = re.sub(r'[^\w\s-]', '', text)  # Спецсимволы
        text = re.sub(r'\d+', '', text)  # Цифры
        text = text.lower().strip()  # Нормализация регистра

        words = []
        for word in text.split():
            parsed = morph.parse(word)
            if parsed:
                lemma = parsed[0].normal_form
                if lemma not in russian_stopwords and len(lemma) > 2:
                    words.append(lemma)

        return ' '.join(words)

    return [clean_text(text) for text in texts if text.strip()]


def get_vectors(texts: list[str]) -> list:
    texts = preprocess_texts(texts)
    vectors = []
    if not is_cache_not_empty():
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        for idx, text in enumerate(texts):
            vector = model.encode(text)
            vectors.append(vector)
            print(idx, len(vector), vector[:5])
        else:
            save_to_cache(vectors)
    else:
        vectors = load_from_cache()

    return vectors


def get_cluster_keywords(texts, labels, top_n=10, max_features=5000):
    clustered_texts = {}
    for text, label in zip(texts, labels):
        clustered_texts.setdefault(label, []).append(text)

    cluster_keywords = {}
    for label, docs in clustered_texts.items():
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            token_pattern=r'(?u)\b[a-zA-Zа-яА-ЯёЁ]{3,}\b'
        )

        tfidf_matrix = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()

        scores = tfidf_matrix.sum(axis=0).A1
        top_indices = scores.argsort()[-top_n:][::-1]

        keywords = [feature_names[i] for i in top_indices]
        cluster_keywords[label] = keywords

        word_freq = {feature_names[i]: scores[i] for i in top_indices}
        wc = WordCloud(
            font_path='arial'
        ).generate_from_frequencies(word_freq)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(f'Кластер {label} ({len(docs)} документов)')
        plt.axis('off')
        plt.show()


def hierarchical_clustering(vectors, n_clusters=2, method='ward', metric='euclidean'):
    """
    Выполняет иерархическую кластеризацию для массива векторов

    Параметры:
    vectors (list/np.array): Массив векторов формы (n_samples, n_features)
    n_clusters (int): Количество желаемых кластеров
    method (str): Метод связывания ('ward', 'complete', 'average', 'single')
    metric (str): Метрика расстояния ('euclidean', 'cosine', 'cityblock' и др.)

    Возвращает:
    np.array: Массив меток кластеров формы (n_samples,)
    """
    # Преобразование в numpy array
    X = np.array(vectors)

    # Проверка размерности данных
    if X.ndim != 2:
        raise ValueError("Input data must be 2-dimensional array (n_samples, n_features)")

    # Вычисление попарных расстояний
    distance_matrix = pdist(X, metric=metric)

    # Построение иерархии кластеров
    Z = linkage(distance_matrix, method=method)

    # Формирование кластеров
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    return labels


def number_of_clusters(vectors):
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(vectors)
        wcss.append(kmeans.inertia_)

    kl = KneeLocator(
        range(1, 11),
        wcss,
        curve='convex',
        direction='decreasing'
    )
    return kl.elbow


def kmeans_clustering(vectors, n_clusters, group_ids):
    vectors = np.asarray(vectors)
    group_ids = np.asarray(group_ids)

    kmeans = KMeans(n_clusters=n_clusters, n_init='auto').fit(vectors)
    labels = kmeans.labels_
    adjusted_labels = np.copy(labels)

    for cluster in np.unique(labels):
        cluster_mask = (labels == cluster)
        cluster_groups = group_ids[cluster_mask]

        unique_groups, counts = np.unique(cluster_groups, return_counts=True)
        predominant_group = unique_groups[np.argmax(counts)]

        non_predominant_mask = cluster_mask & (group_ids != predominant_group)
        adjusted_labels[non_predominant_mask] = -1

    return adjusted_labels, kmeans.cluster_centers_


def dbscan_clustering(vectors, eps=2.00, min_samples=768):
    """
    Кластеризация 768-мерных векторов методом DBSCAN

    Параметры:
        vectors : np.array - массив векторов формы (n_samples, 768)
        eps : float - максимальное расстояние между соседями
        min_samples : int - минимальное число образцов в окрестности

    Возвращает:
        tuple (метки кластеров, число кластеров)
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(vectors)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters


def make_3d_plot(vectors, texts, labels, group_ids):
    pca = PCA(n_components=3)
    vectors_3d = pca.fit_transform(vectors)

    df = pd.DataFrame({
        'text': [text[:150] for text in texts],
        'cluster': labels,
        'group id': group_ids,
        'x': vectors_3d[:, 0],
        'y': vectors_3d[:, 1],
        'z': vectors_3d[:, 2]
    })

    fig = px.scatter_3d(df,
                        x='x', y='y', z='z',
                        color='cluster',
                        hover_data=['text', 'group id'],
                        title='3D Визуализация кластеров',
                        labels={'cluster': 'Кластер'},
                        color_continuous_scale=px.colors.sequential.Viridis)

    fig.update_layout(
        scene=dict(
            xaxis_title='PCA 1',
            yaxis_title='PCA 2',
            zaxis_title='PCA 3'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    fig.show()


def make_2d_plot(vectors, texts, labels, group_ids):
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    df_2d = pd.DataFrame({
        'text': [text[:150] for text in texts],
        'cluster': labels,
        'group id': group_ids,
        'x': vectors_2d[:, 0],
        'y': vectors_2d[:, 1]
    })

    fig = px.scatter(df_2d,
                     x='x',
                     y='y',
                     color='cluster',
                     hover_data=['text', 'group id'],
                     title='2D Визуализация кластеров',
                     labels={'cluster': 'Кластер'},
                     color_continuous_scale=px.colors.sequential.Viridis)

    fig.update_layout(
        xaxis_title='PCA 1',
        yaxis_title='PCA 2',
        margin=dict(l=0, r=0, b=0, t=30)
    )
    fig.show()
