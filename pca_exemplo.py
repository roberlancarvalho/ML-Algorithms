from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

X_complexo = np.array([
    [10.2, 5.1, 8.3, 2.0, 1.5],
    [11.5, 6.0, 9.1, 2.5, 1.8],
    [9.8, 4.8, 7.9, 1.8, 1.3],
    [12.1, 6.3, 9.7, 2.7, 2.0],
    [10.0, 5.0, 8.0, 1.9, 1.4]
])

scaler = StandardScaler()
X_std = scaler.fit_transform(X_complexo)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

print("Dados transformados em 2 componentes:")
print(X_pca)

print("\nVariância explicada por componente:")
print(pca.explained_variance_ratio_)

print("\nVariância total explicada:")
print(sum(pca.explained_variance_ratio_))