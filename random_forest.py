from sklearn.ensemble import RandomForestClassifier

# Dados: [Idade, Score Motor, Sensibilidade]
X = [
    [25, 40, 1],
    [70, 10, 0],
    [45, 30, 1],
    [30, 35, 1],
    [65, 15, 0],
    [50, 20, 0],
    [38, 28, 1]
]

y = [1, 0, 1, 1, 0, 0, 1]

floresta = RandomForestClassifier(n_estimators=100, random_state=42)
floresta.fit(X, y)

novo_paciente = [[50, 35, 1]]
previsao = floresta.predict(novo_paciente)
prognostico = "Recuperação Alta" if previsao[0] == 1 else "Recuperação Baixa"

print(f"Previsão: {previsao[0]}")
print(f"Prognóstico: {prognostico}")

importancias = floresta.feature_importances_
nomes_caracteristicas = ["Idade", "Score Motor", "Sensibilidade"]

print("\nImportância das características:")
for nome, imp in zip(nomes_caracteristicas, importancias):
    print(f"{nome}: {imp:.4f}")