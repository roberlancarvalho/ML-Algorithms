import xgboost as xgb
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

dtrain = xgb.DMatrix(X, label=y)

params = {
    "objective": "binary:logistic",
    "max_depth": 3,
    "eta": 0.1,
    "eval_metric": "auc"
}

bst = xgb.train(params, dtrain, num_boost_round=20)

novo = np.array([[50, 35, 1]])
prob = bst.predict(xgb.DMatrix(novo))

print(f"Probabilidade de Recuperação Alta: {prob[0]:.2%}")

classe = 1 if prob[0] >= 0.5 else 0
print("Classe prevista:", classe)