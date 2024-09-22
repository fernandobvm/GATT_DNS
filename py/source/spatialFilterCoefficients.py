import numpy as np

class SpatialFilterCoefficients:
    def __init__(self, alpha, filter_borders):
        self.alpha = alpha
        self.filter_borders = filter_borders
        self.filter_stencil_LHS = [1, alpha]
        self.filter_stencil_RHS = [
            (193 + 126 * alpha) / 256, (105 + 302 * alpha) / 512, 
            (-15 + 30 * alpha) / 128, (45 - 90 * alpha) / 1024, 
            (-5 + 10 * alpha) / 512, (1 - 2 * alpha) / 1024
        ]
        self.filter_decentered_stencil_LHS = None
        self.filter_decentered_stencil_RHS = None
        
        self.prepare_filter_borders()

    def prepare_filter_borders(self):
        if isinstance(self.filter_borders, bool):
            self.filter_borders = 'decentered' if self.filter_borders else 'off'

        if self.filter_borders == 'decentered':
            self.filter_decentered_stencil_LHS = [1, self.alpha]
            self.filter_decentered_stencil_RHS = np.zeros((11, 5))

            a_bound = np.zeros((11, 5))

            a_bound[0, 4] = (-1 + 2 * self.alpha) / 1024
            a_bound[1, 4] = (5 - 10 * self.alpha) / 512
            a_bound[2, 4] = (-45 + 90 * self.alpha) / 1024
            a_bound[3, 4] = (15 + 98 * self.alpha) / 128
            a_bound[4, 4] = (407 + 210 * self.alpha) / 512
            a_bound[5, 4] = (63 + 130 * self.alpha) / 256
            a_bound[6, 4] = (-105 + 210 * self.alpha) / 512
            a_bound[7, 4] = (15 - 30 * self.alpha) / 128
            a_bound[8, 4] = (-45 + 90 * self.alpha) / 1024
            a_bound[9, 4] = (5 - 10 * self.alpha) / 512
            a_bound[10, 4] = (-1 + 2 * self.alpha) / 1024

            a_bound[0, 3] = (1 - 2 * self.alpha) / 1024
            a_bound[1, 3] = (-5 + 10 * self.alpha) / 512
            a_bound[2, 3] = (45 + 934 * self.alpha) / 1024
            a_bound[3, 3] = (113 + 30 * self.alpha) / 128
            a_bound[4, 3] = (105 + 302 * self.alpha) / 512
            a_bound[5, 3] = (-63 + 126 * self.alpha) / 256
            a_bound[6, 3] = (105 - 210 * self.alpha) / 512
            a_bound[7, 3] = (-15 + 30 * self.alpha) / 128
            a_bound[8, 3] = (45 - 90 * self.alpha) / 1024
            a_bound[9, 3] = (-5 + 10 * self.alpha) / 512
            a_bound[10, 3] = (1 - 2 * self.alpha) / 1024

            a_bound[0, 2] = (-1 + 2 * self.alpha) / 1024
            a_bound[1, 2] = (5 + 502 * self.alpha) / 512
            a_bound[2, 2] = (979 + 90 * self.alpha) / 1024
            a_bound[3, 2] = (15 + 98 * self.alpha) / 128
            a_bound[4, 2] = (-105 + 210 * self.alpha) / 512
            a_bound[5, 2] = (63 - 126 * self.alpha) / 256
            a_bound[6, 2] = (-105 + 210 * self.alpha) / 512
            a_bound[7, 2] = (15 - 30 * self.alpha) / 128
            a_bound[8, 2] = (-45 + 90 * self.alpha) / 1024
            a_bound[9, 2] = (5 - 10 * self.alpha) / 512
            a_bound[10, 2] = (-1 + 2 * self.alpha) / 1024

            a_bound[0, 1] = (1 + 1022 * self.alpha) / 1024
            a_bound[1, 1] = (507 + 10 * self.alpha) / 512
            a_bound[2, 1] = (45 + 934 * self.alpha) / 1024
            a_bound[3, 1] = (-15 + 30 * self.alpha) / 128
            a_bound[4, 1] = (105 - 210 * self.alpha) / 512
            a_bound[5, 1] = (-63 + 126 * self.alpha) / 256
            a_bound[6, 1] = (105 - 210 * self.alpha) / 512
            a_bound[7, 1] = (-15 + 30 * self.alpha) / 128
            a_bound[8, 1] = (45 - 90 * self.alpha) / 1024
            a_bound[9, 1] = (-5 + 10 * self.alpha) / 512
            a_bound[10, 1] = (1 - 2 * self.alpha) / 1024

            a_bound[0, 0] = (1023 + 1 * self.alpha) / 1024
            a_bound[1, 0] = (5 + 507 * self.alpha) / 512
            a_bound[2, 0] = (-45 + 45 * self.alpha) / 1024
            a_bound[3, 0] = (15 - 15 * self.alpha) / 128
            a_bound[4, 0] = (-105 + 105 * self.alpha) / 512
            a_bound[5, 0] = (63 - 63 * self.alpha) / 256
            a_bound[6, 0] = (-105 + 105 * self.alpha) / 512
            a_bound[7, 0] = (15 - 15 * self.alpha) / 128
            a_bound[8, 0] = (-45 + 45 * self.alpha) / 1024
            a_bound[9, 0] = (5 - 5 * self.alpha) / 512
            a_bound[10, 0] = (-1 + 1 * self.alpha) / 1024

            self.filter_decentered_stencil_RHS = a_bound.T

        elif self.filter_borders == 'reducedOrder':
            self.filter_decentered_stencil_LHS = 1

            F2 = [(1/2 + self.alpha) / 1, (1/2 + self.alpha) / 2]
            F4 = [(5/8 + 3/4 * self.alpha) / 1, (1/2 + self.alpha) / 2, (-1/8 + 1/4 * self.alpha) / 2]
            F6 = [(11/16 + 5/8 * self.alpha) / 1, (15/32 + 17/16 * self.alpha) / 2, (-3/16 + 3/8 * self.alpha) / 2, (1/32 - 1/16 * self.alpha) / 2]
            F8 = [(93/128 + 70/128 * self.alpha) / 1, (7/16 + 18/16 * self.alpha) / 2, (-7/32 + 14/32 * self.alpha) / 2, (1/16 - 1/8 * self.alpha) / 2, (-1/128 + 1/64 * self.alpha) / 2]

            self.filter_decentered_stencil_RHS = np.zeros((5, 9))
            self.filter_decentered_stencil_RHS[0, 0] = 1
            self.filter_decentered_stencil_RHS[1, :3] = F2
            self.filter_decentered_stencil_RHS[2, :5] = F4
            self.filter_decentered_stencil_RHS[3, :7] = F6
            self.filter_decentered_stencil_RHS[4, :9] = F8

        elif self.filter_borders == 'off':
            self.filter_decentered_stencil_LHS = np.diag(np.ones(5))
            self.filter_decentered_stencil_RHS = np.diag(np.ones(5))


# Exemplo de uso
alpha = 0.5  # Exemplo de valor para alpha
filter_borders = 'decentered'  # Exemplo de configuração para filter_borders ('decentered', 'reducedOrder', ou 'off')

# Criando um objeto da classe SpatialFilterCoefficients
spatial_filter = SpatialFilterCoefficients(alpha, filter_borders)

# Acessando os resultados
print("Filter Stencil LHS:", spatial_filter.filter_stencil_LHS)
print("Filter Stencil RHS:", spatial_filter.filter_stencil_RHS)

if spatial_filter.filter_borders == 'decentered':
    print("Filter Decentered Stencil LHS:", spatial_filter.filter_decentered_stencil_LHS)
    print("Filter Decentered Stencil RHS:", spatial_filter.filter_decentered_stencil_RHS)
elif spatial_filter.filter_borders == 'reducedOrder':
    print("Filter Reduced Order Stencil LHS:", spatial_filter.filter_decentered_stencil_LHS)
    print("Filter Reduced Order Stencil RHS:", spatial_filter.filter_decentered_stencil_RHS)
elif spatial_filter.filter_borders == 'off':
    print("Filter Off Stencil LHS:", spatial_filter.filter_decentered_stencil_LHS)
    print("Filter Off Stencil RHS:", spatial_filter.filter_decentered_stencil_RHS)
 