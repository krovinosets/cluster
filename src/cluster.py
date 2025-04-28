import os

import psycopg2
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


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
    for idx, text in enumerate(texts):
        vector = model.encode(text)
        vectors.append(vector)
        print(idx, len(vector), vector[:5])

    min_samples = 1536
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(vectors)
    distances, indices = neighbors_fit.kneighbors(vectors)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{min_samples}-NN distance')
    plt.show()

    knee_index = 3700  # примерная позиция изгиба
    eps = distances[knee_index]
    print(eps)

    # Кластеризация DBSCAN
    clustering = DBSCAN(eps=2.00, min_samples=min_samples).fit(vectors)
    labels = clustering.labels_

    # Уменьшение размерности для визуализации
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
