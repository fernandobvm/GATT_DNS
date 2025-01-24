import sys
import numpy as np
import matplotlib.pyplot as plt
from auxFunctions import func_loadFlow  # Função externa
import cmcrameri.cm as cm
from matplotlib.colors import ListedColormap


if __name__ == "__main__":
    case_name = "baseflow_gap_Re950_M01_h10_LR18"  # Default case name

    # Check if a case name was passed as a command-line argument
    if len(sys.argv) > 1:
        case_name = sys.argv[1]

    print(f"Using case name: {case_name}")
# Configurações iniciais
dir_path = "../"

ax_lim = [320.82 - 20, 320.82 + 20, -np.inf, 0.1]  # Limites do domínio plotado
# ax_lim = [320.82 - 5, 320.82 + 5, -np.inf, 0.1]
# ax_lim = [311.6, 312.2, -np.inf, 0.1]

var = "U"  # Variável de plotagem: 'U', 'V', 'R', 'E'
steps = None  # Número do arquivo qSave ou último qSave se None
d_step = None  # Delta qSave (para visualização da evolução temporal)
n_step = 100  # Número de qSave (últimos)

ifig = 1  # Índice da figura
c_lim = None  # Limites de cor (vazio: valores [min, max])
n_lvls = 10  # Número de níveis de contorno
c_map = ListedColormap(cm.davos(np.linspace(0, 1, n_lvls)))  # Definição do mapa de cores

# Importar dados e plotar
x, y, t, flow = func_loadFlow(f"{dir_path}{case_name}", ax_lim, [var.upper()], steps, d_step, n_step)

# Chamada da função de plotagem
p_contourf(x, y, t, var, flow[var], ifig, c_lim, c_map)


def p_contourf(x, y, t, var, flow, ifig, c_lim, c_map):
    if c_lim is None:
        c_lim = [np.min(flow), np.max(flow)]

    c_lvls = np.linspace(c_lim[0], c_lim[1], len(c_map) + 1)

    plt.figure(ifig)
    plt.clf()
    ax = plt.gca()
    ax.set_position([0.12, 0.1, 0.85, 0.75])
    ax.set_xlim(ax_lim[0], ax_lim[1])
    ax.set_ylim(ax_lim[2], ax_lim[3])
    
    im = ax.contourf(x, y, flow[:, :, 0].T, levels=c_lvls, cmap=c_map)
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal")
    cbar.ax.set_position([ax.get_position().x0, ax.get_position().y1 + 0.05, ax.get_position().width, 0.04])
    cbar.set_label(f'${var}$', fontsize=15)

    ax.set_xlabel("$x$", fontsize=15)
    ax.set_ylabel("$y$", fontsize=15)
    ax.grid(True)

    if len(t) > 1:
        for it in range(1, len(t)):
            ax.contourf(x, y, flow[:, :, it].T, levels=c_lvls, cmap=c_map)
            plt.pause(1)

    plt.show()
