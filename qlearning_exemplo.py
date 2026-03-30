import numpy as np

num_estados = 5
num_acoes = 3

q_table = np.zeros((num_estados, num_acoes))

alpha = 0.1
gamma = 0.9

def get_reward(estabilidade, esforco):
    if estabilidade > 0.8:
        return 10
    return -5

estado = 0
acao = 1
proximo_estado = 2

recompensa = get_reward(0.9, 0.5)

erro_previsao = recompensa + gamma * np.max(q_table[proximo_estado]) - q_table[estado, acao]
q_table[estado, acao] += alpha * erro_previsao

print("Tabela Q atualizada:")
print(q_table)