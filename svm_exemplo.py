from sklearn.svm import SVC
import numpy as np

# Dados já reduzidos para 2 componentes
X_pca = np.array([
    [1.0, 2.0],
    [1.5, 2.5],
    [1.2, 3.5],
    [1.8, 3.0],
    [3.5, 1.0],
    [4.0, 1.5],
    [3.8, 0.8],
    [4.2, 1.2]
])

# 1 = marcha independente, 0 = marcha assistida
y = np.array([1, 1, 1, 1, 0, 0, 0, 0])

modelo_svm = SVC(kernel="rbf", C=1.0, gamma="scale")
modelo_svm.fit(X_pca, y)

vetores_suporte = modelo_svm.support_vectors_
print(f"Total de vetores de suporte: {len(vetores_suporte)}")

novo_caso_pca = np.array([[2.0, 2.8]])
previsao = modelo_svm.predict(novo_caso_pca)

prognostico = "Marcha Independente" if previsao[0] == 1 else "Marcha Assistida"
print(f"Prognóstico: {prognostico}")