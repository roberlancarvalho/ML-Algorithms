from sklearn.tree import DecisionTreeClassifier
import numpy as np

X = np.array([
    [25, 40, 1],
    [70, 10, 0],
    [45, 30, 1],
    [30, 35, 1],
    [65, 15, 0],
    [50, 20, 0],
    [38, 28, 1]
])

y = np.array([1, 0, 1, 1, 0, 0, 1])

modelo = DecisionTreeClassifier(max_depth=2, random_state=42)
modelo.fit(X, y)

novo_paciente = np.array([[50, 35, 1]])
previsao = modelo.predict(novo_paciente)

prognostico = "Recuperação Alta" if previsao[0] == 1 else "Recuperação Baixa"
print(f"Previsão: {previsao[0]}")
print(f"Prognóstico para o novo paciente: {prognostico}")

if hasattr(modelo, "predict_proba"):
    prob = modelo.predict_proba(novo_paciente)
    print("Probabilidades:", prob)