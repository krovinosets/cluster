import os

import psycopg2
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
from kneed import KneeLocator
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from src.cache import save_to_cache, is_cache_not_empty, load_from_cache


def get_vk_posts_texts() -> list[str]:
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
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT text FROM vk_posts;")
            results = cursor.fetchall()
            texts = [row[0] for row in results]
    except psycopg2.Error as e:
        print(f"Ошибка при выполнении запроса: {e}")

    return texts


def make_dbscan(texts: list[str]) -> None:
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    vectors = []
    if not is_cache_not_empty():
        for idx, text in enumerate(texts):
            vector = model.encode(text)
            vectors.append(vector)
            print(idx, len(vector), vector[:5])
        else:
            save_to_cache(vectors)
    else:
        vectors = load_from_cache()

    min_samples = 1536
    # neighbors = NearestNeighbors(n_neighbors=min_samples)
    # neighbors_fit = neighbors.fit(vectors)
    # distances, indices = neighbors_fit.kneighbors(vectors)
    # distances = np.sort(distances, axis=0)
    # distances = distances[:, 1]
    # plt.plot(distances)
    # plt.xlabel('Points sorted by distance')
    # plt.ylabel(f'{min_samples}-NN distance')
    # plt.show()
    #
    # knee_index = 3700  # примерная позиция изгиба
    # eps = distances[knee_index]
    # print(eps)

    # labels, n_clusters = dbscan_clustering(vectors)
    # print(n_clusters)
    #
    # # # Кластеризация DBSCAN
    # # clustering = DBSCAN(eps=2.00, min_samples=min_samples).fit(vectors)
    # # labels = clustering.labels_
    #
    # make_2d_plot(vectors, texts, labels)

    num = number_of_clusters(vectors)
    labels, centers = kmeans_clustering(vectors, num)
    make_2d_plot(vectors, texts, labels)
    make_3d_plot(vectors, texts, labels)


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
    optimal_k = kl.elbow
    return optimal_k


def kmeans_clustering(vectors, n_clusters):
    """
    Кластеризация 768-мерных векторов методом K-means

    Параметры:
        vectors : np.array - массив векторов формы (n_samples, 768)
        n_clusters : int - количество кластеров

    Возвращает:
        tuple (метки кластеров, центроиды кластеров)
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto').fit(vectors)
    return kmeans.labels_, kmeans.cluster_centers_


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


def make_3d_plot(vectors, texts, labels):
    pca = PCA(n_components=3)
    vectors_3d = pca.fit_transform(vectors)

    # Создание DataFrame для визуализации
    df = pd.DataFrame({
        'text': [text[:150] for text in texts],
        'cluster': labels,
        'x': vectors_3d[:, 0],
        'y': vectors_3d[:, 1],
        'z': vectors_3d[:, 2]
    })

    # Интерактивная 3D визуализация
    fig = px.scatter_3d(df,
                        x='x', y='y', z='z',
                        color='cluster',
                        hover_data=['text'],
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


def make_2d_plot(vectors, texts, labels):
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    # Создание DataFrame
    df_2d = pd.DataFrame({
        'text': [text[:150] for text in texts],
        'cluster': labels,
        'x': vectors_2d[:, 0],
        'y': vectors_2d[:, 1]
    })

    # Интерактивная 2D визуализация
    fig = px.scatter(df_2d,
                     x='x',
                     y='y',
                     color='cluster',
                     hover_data=['text'],
                     title='2D Визуализация кластеров',
                     labels={'cluster': 'Кластер'},
                     color_continuous_scale=px.colors.sequential.Viridis)

    fig.update_layout(
        xaxis_title='PCA 1',
        yaxis_title='PCA 2',
        margin=dict(l=0, r=0, b=0, t=30)
    )
    fig.show()
