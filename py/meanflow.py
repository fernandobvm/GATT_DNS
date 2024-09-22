import numpy as np
import os

# Definir os parâmetros de entrada e saída
Tin = np.arange(11700, 16601)  # O último valor no MATLAB é inclusivo
pasta_in = 'rugosidade6_5_BC2'
Tout = Tin[-1] + 1
pasta_out = pasta_in

# Inicializar as variáveis para acumular os dados
U = V = W = R = E = None

# Iterar pelos arquivos especificados em Tin
for t in Tin:
    tstr = os.path.join(pasta_in, f'flow_{t:010d}.npz')
    print(f'Carregando arquivo: {tstr}')
    
    # Carregar os arquivos .npz
    current = np.load(tstr)
    
    # Se for o primeiro arquivo, inicializa as variáveis
    if t == Tin[0]:
        U = current['U']
        V = current['V']
        W = current['W']
        R = current['R']
        E = current['E']
    else:
        # Somar os valores dos arquivos subsequentes
        U += current['U']
        V += current['V']
        W += current['W']
        R += current['R']
        E += current['E']

# Fazer a média dos valores acumulados
U /= len(Tin)
V /= len(Tin)
W /= len(Tin)
R /= len(Tin)
E /= len(Tin)

# Nome do arquivo de saída
tstr_out = os.path.join(pasta_out, f'flow_{Tout:010d}.npz')

# Salvar os dados acumulados em formato .npz
np.savez(tstr_out, t=current['t'], U=U, V=V, W=W, R=R, E=E)

print(f'Dados salvos em {tstr_out}')
