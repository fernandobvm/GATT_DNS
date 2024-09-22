import numpy as np

def get_domain_slices(n, p):
    """
    Esta função calcula o tamanho de cada fatia da mesma forma que a biblioteca 2decomp faz.
    É usada para distribuir corretamente as informações entre os processos.
    A distribuição é a mais equilibrada possível. Se for necessário uma distribuição desigual,
    os nós extras são colocados nas últimas fatias.

    Por exemplo, se 10 nós forem divididos em 3 fatias, a divisão seria: 3, 3, 4.
    """

    # Caso especial: se n for 1, retornamos uma fatia única
    if n == 1:
        return np.array([[1], [1]])

    # Cálculo do número base de pontos por fatia e do número de fatias maiores
    n_points_base = n // p
    n_ceil = n - n_points_base * p

    # Inicializamos todas as fatias com o valor base
    n_points = np.ones(p, dtype=int) * n_points_base

    # Atribuímos fatias maiores para os últimos nós, se necessário
    n_points[-n_ceil:] = n_points_base + 1

    # Calculamos o índice final de cada fatia
    slices = np.zeros((2, p), dtype=int)
    slices[1, :] = np.cumsum(n_points)

    # O índice inicial da primeira fatia é 1, e os demais seguem a sequência
    slices[0, 0] = 1
    slices[0, 1:] = slices[1, :-1] + 1

    return slices