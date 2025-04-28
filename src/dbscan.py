from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from keybert import KeyBERT

texts = [
    # Техника (4)
    "Смартфон быстро разряжается, хотя новый",
    "Ноутбук греется при играх, нужно чистить",
    "Планшет тормозит после обновления, не нравится",
    "Наушники отличные, звук просто вау!",

    # Кино (5)
    "Концовка фильма предсказуемая, разочарован",
    "Актеры играют неестественно, не верю",
    "Спецэффекты как в голливуде, круто!",
    "Сюжет скучный, засыпал на середине",
    "Лучший фильм года, пересматриваю третий раз",

    # Книги (4)
    "Стиль автора сложный, тяжело читать",
    "Сюжет затянут, могло быть короче",
    "Идея оригинальная, такого еще не было",
    "Персонажи плоские, не запоминаются",

    # Смешанные (3)
    "Норм фильм, но затянуто",
    "В целом неплохо, но есть минусы",
    "Ничего особенного, обычный продукт"
]

# Инициализация моделей
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
sentiment_analyzer = pipeline('sentiment-analysis', model='blanchefort/rubert-base-cased-sentiment')
kw_model = KeyBERT(model='paraphrase-multilingual-mpnet-base-v2')


# Анализ тональности
def analyze_sentiment(texts):
    results = sentiment_analyzer(texts)
    return [1 if res['label'] == 'POSITIVE' else -1 if res['label'] == 'NEGATIVE' else 0 for res in results]


# Генерация признаков
embeddings = model.encode(texts)
sentiments = np.array(analyze_sentiment(texts)).reshape(-1, 1)
combined_features = np.hstack([embeddings, sentiments * 0.5])

# Кластеризация
clustering = DBSCAN(
    eps=0.52,  # Увеличенный радиус
    min_samples=3,  # Минимум 3 комментария в кластере
    metric='cosine'
).fit(combined_features)
tsne = TSNE(n_components=2, perplexity=5, learning_rate=200, n_iter=2500, random_state=42)
projections = tsne.fit_transform(embeddings)

# Визуализация с цветовой кодировкой
plt.figure(figsize=(14, 10))
for label in set(clustering.labels_):
    if label == -1: continue
    mask = clustering.labels_ == label
    plt.scatter(projections[mask, 0], projections[mask, 1],
                cmap='viridis', s=100, label=f'Cluster {label}',
                edgecolors='w', linewidth=0.5)

# Аннотации с тональностью
for i, (txt, sent) in enumerate(zip(texts, sentiments)):
    color = 'green' if sent > 0 else 'red' if sent < 0 else 'gray'
    plt.annotate(txt, (projections[i, 0], projections[i, 1]),
                 fontsize=9, color=color,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color, lw=0.5))

plt.title("Анализ кластеров по тематике и оценкам")
plt.legend()
plt.show()

# Анализ результатов
clusters = {}
for idx, label in enumerate(clustering.labels_):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(texts[idx])

print("\nДетализация кластеров:")
for label, items in sorted(clusters.items(), key=lambda x: x[0]):
    if label == -1:
        print(f"\nШумовые точки ({len(items)}):")
    else:
        cluster_texts = [texts[i] for i in np.where(clustering.labels_ == label)[0]]
        pos_ratio = sum(1 for s in sentiments[clustering.labels_ == label] if s > 0) / len(cluster_texts)
        tone = "Позитивный" if pos_ratio > 0.6 else "Негативный" if pos_ratio < 0.4 else "Смешанный"

        keywords = kw_model.extract_keywords('\n'.join(cluster_texts),
                                             keyphrase_ngram_range=(1, 2),
                                             stop_words=None,
                                             top_n=3)

        print(f"\nКластер {label} ({tone}, {len(items)} коммент.):")
        print(f"Ключевые темы: {', '.join([kw[0] for kw in keywords])}")
        print("Примеры:")
        for txt in items[:3]:
            print(f" - {txt}")

# Статистика
print("\nСтатистика:")
print(f"Всего кластеров: {len(clusters) - (1 if -1 in clusters else 0)}")
print(f"Не распределено: {len(clusters.get(-1, []))} ({len(clusters.get(-1, [])) / len(texts):.0%})")