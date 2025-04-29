from kneed import KneeLocator

from src.cache import load_from_cache

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = load_from_cache()
print(len(X))

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


from kneed import KneeLocator

kl = KneeLocator(
    range(1, 11),
    wcss,
    curve='convex',
    direction='decreasing'
)
optimal_k = kl.elbow
print(f"Оптимальное число кластеров: {optimal_k}")