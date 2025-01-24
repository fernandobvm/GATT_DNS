import os
from copy import deepcopy
import numpy as np
from source.library import custom_range, Disturbance


def init_boundaries(boundary, mesh, domainSlicesY, domainSlicesZ, p_row, p_col):
    biG = BoundaryInfo(boundary, mesh, domainSlicesY, domainSlicesZ, p_row, p_col)

    # Ordem das direções
    direction_order = ['xi', 'xf', 'yi', 'yf', 'zi', 'zf']

    for i in range(len(boundary.val)):
        if boundary.type[i] == 'dir':
            if boundary.var[i] == 'u':
                biG.nUd += 1
                biG.iUd.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                biG.vUd.append(boundary.val[i])
            elif boundary.var[i] == 'v':
                biG.nVd += 1
                biG.iVd.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                biG.vVd.append(boundary.val[i])
            elif boundary.var[i] == 'w':
                biG.nWd += 1
                biG.iWd.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                biG.vWd.append(boundary.val[i])
            elif boundary.var[i] == 'p':
                biG.nPd += 1
                biG.iPd.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                biG.vPd.append(boundary.val[i])
            elif boundary.var[i] == 'e':
                biG.nEd += 1
                biG.iEd.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                biG.vEd.append(boundary.val[i])
        elif boundary.type[i] == 'neu':
            direction_index = direction_order.index(boundary.dir[i])
            if boundary.var[i] == 'u':
                biG.nUn += 1
                biG.iUn.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                biG.dUn.append(direction_index+1)
            elif boundary.var[i] == 'v':
                biG.nVn += 1
                biG.iVn.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                biG.dVn.append(direction_index+1)
            elif boundary.var[i] == 'w':
                biG.nWn += 1
                biG.iWn.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                biG.dWn.append(direction_index+1)
            elif boundary.var[i] == 'p':
                biG.nPn += 1
                biG.iPn.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                biG.dPn.append(direction_index+1)
            elif boundary.var[i] == 'e':
                biG.nEn += 1
                biG.iEn.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                biG.dEn.append(direction_index+1)
        elif boundary.type[i] == 'sec':
            direction_index = direction_order.index(boundary.dir[i])
            if boundary.var[i] == 'u':
                biG.nUs += 1
                biG.iUs.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                biG.dUs.append(direction_index+1)
            elif boundary.var[i] == 'v':
                biG.nVs += 1
                biG.iVs.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                biG.dVs.append(direction_index+1)
            elif boundary.var[i] == 'w':
                biG.nWs += 1
                biG.iWs.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                biG.dWs.append(direction_index+1)
            elif boundary.var[i] == 'p':
                biG.nPs += 1
                biG.iPs.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                biG.dPs.append(direction_index+1)
            elif boundary.var[i] == 'e':
                biG.nEs += 1
                biG.iEs.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                biG.dEs.append(direction_index+1)

    # Configurações dos corners
    biG.cL = boundary.corners.limits
    biG.cD = boundary.corners.dir
    biG.adiabatic = boundary.corners.adiabatic
    biG.cN = len(biG.cL)

    # Coeficientes de Neumann
    neumann_length = len(boundary.neumann_coeffs)
    neumann2_length = len(boundary.neumann2_coeffs)

    # Valor de gamma-1 para conversão de pressão em densidade
    biG.gamma1 = boundary.gamma - 1

    # Configurações do campo elétrico
    biG.E0 = boundary.E0

    # Divisão para diferentes processadores
    bi = [None] * (p_row * p_col)
    for j in range(p_row):
        for k in range(p_col):
            n_proc = k + j * p_col
            biL = deepcopy(biG)  # Cria cópia temporária para este processador

            Ji = domainSlicesY[0, j]
            Jf = domainSlicesY[1, j]
            Ki = domainSlicesZ[0, k]
            Kf = domainSlicesZ[1, k]

            # Limitar índices (função externa)
            biL.iUd, biL.nUd, biL.vUd = biG.limit_indices(biL.iUd, biL.nUd, biL.vUd, 'd', Ji, Jf, Ki, Kf, 0)
            biL.iVd, biL.nVd, biL.vVd = biG.limit_indices(biL.iVd, biL.nVd, biL.vVd, 'd', Ji, Jf, Ki, Kf, 0)
            biL.iWd, biL.nWd, biL.vWd = biG.limit_indices(biL.iWd, biL.nWd, biL.vWd, 'd', Ji, Jf, Ki, Kf, 0)
            biL.iPd, biL.nPd, biL.vPd = biG.limit_indices(biL.iPd, biL.nPd, biL.vPd, 'd', Ji, Jf, Ki, Kf, 0)
            biL.iEd, biL.nEd, biL.vEd = biG.limit_indices(biL.iEd, biL.nEd, biL.vEd, 'd', Ji, Jf, Ki, Kf, 0)

            biL.iUn, biL.nUn, biL.dUn = biG.limit_indices(biL.iUn, biL.nUn, biL.dUn, 'n', Ji, Jf, Ki, Kf, neumann_length)
            biL.iVn, biL.nVn, biL.dVn = biG.limit_indices(biL.iVn, biL.nVn, biL.dVn, 'n', Ji, Jf, Ki, Kf, neumann_length)
            biL.iWn, biL.nWn, biL.dWn = biG.limit_indices(biL.iWn, biL.nWn, biL.dWn, 'n', Ji, Jf, Ki, Kf, neumann_length)
            biL.iPn, biL.nPn, biL.dPn = biG.limit_indices(biL.iPn, biL.nPn, biL.dPn, 'n', Ji, Jf, Ki, Kf, neumann_length)
            biL.iEn, biL.nEn, biL.dEn = biG.limit_indices(biL.iEn, biL.nEn, biL.dEn, 'n', Ji, Jf, Ki, Kf, neumann_length)

            biL.iUs, biL.nUs, biL.dUs = biG.limit_indices(biL.iUs, biL.nUs, biL.dUs,'n',Ji,Jf,Ki,Kf,neumann2_length)
            biL.iVs, biL.nVs, biL.dVs = biG.limit_indices(biL.iVs, biL.nVs, biL.dVs,'n',Ji,Jf,Ki,Kf,neumann2_length)
            biL.iWs, biL.nWs, biL.dWs = biG.limit_indices(biL.iWs, biL.nWs, biL.dWs,'n',Ji,Jf,Ki,Kf,neumann2_length)
            biL.iPs, biL.nPs, biL.dPs = biG.limit_indices(biL.iPs, biL.nPs, biL.dPs,'n',Ji,Jf,Ki,Kf,neumann2_length)
            biL.iEs, biL.nEs, biL.dEs = biG.limit_indices(biL.iEs, biL.nEs, biL.dEs,'n',Ji,Jf,Ki,Kf,neumann2_length)
            # Similar para as outras variáveis...

            #values = [biL.cD, biL.adiabatic]
            values = np.hstack((np.array(biL.cD), biL.adiabatic))
            biL.cL, biL.cN, values = biG.limit_indices(biL.cL, biL.cN, values, 'c', Ji, Jf, Ki, Kf, neumann_length)
            biL.cD = values[:,0:3]
            biL.adiabatic = values[:, 3]

            # Armazenar resultados no processador
            bi[n_proc] = biL
            
    return split_disturbances(boundary, mesh, domainSlicesY, domainSlicesZ, p_row, p_col, bi)

def split_disturbances(boundary, mesh, domain_slices_y, domain_slices_z, p_row, p_col, bi):

    # Iteração sobre processadores (j e k)
    for j in range(p_row):
        for k in range(p_col):
            n_proc = k + j * p_col  # Calcula o índice do processador atual (Python já está com indexação ajustada)
            disturb = []  # Lista que armazenará os distúrbios para o processador atual
            
            # Itera sobre todos os distúrbios em boundary.disturb
            for i in range(len(boundary.disturb)):
                if boundary.disturb[i] is not None:
                    # ConvMATLAB: O Matlab faz cópia implícita dos vetores nesse tipo de atribuição, enquanto o 
                    # Python apenas compartilha o apontamento. Desse modo, como o vetor sofrerá potencialmente 
                    # múltiplas edições, será preciso fazer uma cópia do vetor.
                    ind = deepcopy(boundary.disturb[i].ind)
                    ind[2:6] = [max(ind[2], domain_slices_y[0][j]), min(ind[3], domain_slices_y[1][j]), 
                                max(ind[4], domain_slices_z[0][k]), min(ind[5], domain_slices_z[1][k])]

                    # Verifica se o índice é válido
                    if ind[2] <= ind[3] and ind[4] <= ind[5]:
                        # Cria um novo objeto Disturbance para adicionar à lista disturb
                        new_disturb = deepcopy(boundary.disturb[i])
                        new_disturb.ind = ind
                        new_disturb.X = mesh.X[ind[0]-1:ind[1]]  # Ajusta o slice para Python
                        new_disturb.Y = mesh.Y[ind[2]-1:ind[3]]
                        new_disturb.Z = mesh.Z[ind[4]-1:ind[5]]
                        
                        disturb.append(new_disturb)
            
            # Atribui a lista de distúrbios ao processador correspondente
            bi[n_proc].disturb = disturb
    return bi

class BoundaryInfo:
    def __init__(self, boundary, mesh, domainSlicesY, domainSlicesZ, p_row, p_col):
        self.nUd = 0
        self.nVd = 0
        self.nWd = 0
        self.nPd = 0
        self.nEd = 0
        self.nUn = 0
        self.nVn = 0
        self.nWn = 0
        self.nPn = 0
        self.nEn = 0
        self.nUs = 0
        self.nVs = 0
        self.nWs = 0
        self.nPs = 0
        self.nEs = 0
        self.iUd = []
        self.iVd = []
        self.iWd = []
        self.iPd = []
        self.iEd = []
        self.iUn = []
        self.iVn = []
        self.iWn = []
        self.iPn = []
        self.iEn = []
        self.iUs = []
        self.iVs = []
        self.iWs = []
        self.iPs = []
        self.iEs = []
        self.vUd = []
        self.vVd = []
        self.vWd = []
        self.vPd = []
        self.vEd = []
        self.dUn = []
        self.dVn = []
        self.dWn = []
        self.dPn = []
        self.dEn = []
        self.dUs = []
        self.dVs = []
        self.dWs = []
        self.dPs = []
        self.dEs = []
        self.cL = None
        self.cD = None
        self.adiabatic = None
        self.cN = 0
        self.gamma1 = 0
        self.E0 = 0

        self.directionOrder = ['xi','xf','yi','yf','zi','zf']
        # Get information on corners
        self.cL = boundary.corners.limits
        self.cD = boundary.corners.dir
        self.adiabatic = boundary.corners.adiabatic
        self.cN = len(self.cL)

        self.neumannLength = len(boundary.neumann_coeffs)
        self.neumann2Length = len(boundary.neumann2_coeffs)
        # Set gamma-1 value for converting pressure to density
        self.gamma1 = boundary.gamma - 1
        # Find regions inside walls
        #self.wR = ~boundary.flowRegion;
        self.E0 = boundary.E0

        self.disturb = []

        #self.__init_boundaries(boundary, mesh, domainSlicesY, domainSlicesZ, p_row, p_col)
        

    def return_vector(self):
        return self.bi


    def __init_boundaries(self, boundary, mesh, domainSlicesY, domainSlicesZ, p_row, p_col):
        #biG = BoundaryInfo(boundary, mesh, domainSlicesY, domainSlicesZ, p_row, p_col)

        # Ordem das direções
        direction_order = ['xi', 'xf', 'yi', 'yf', 'zi', 'zf']

        for i in range(len(boundary.val)):
            if boundary.type[i] == 'dir':
                if boundary.var[i] == 'u':
                    self.nUd += 1
                    self.iUd.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                    self.vUd.append(boundary.val[i])
                elif boundary.var[i] == 'v':
                    self.nVd += 1
                    self.iVd.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                    self.vVd.append(boundary.val[i])
                elif boundary.var[i] == 'w':
                    self.nWd += 1
                    self.iWd.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                    self.vWd.append(boundary.val[i])
                elif boundary.var[i] == 'p':
                    self.nPd += 1
                    self.iPd.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                    self.vPd.append(boundary.val[i])
                elif boundary.var[i] == 'e':
                    self.nEd += 1
                    self.iEd.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                    self.vEd.append(boundary.val[i])
            elif boundary.type[i] == 'neu':
                direction_index = direction_order.index(boundary.dir[i])
                if boundary.var[i] == 'u':
                    self.nUn += 1
                    self.iUn.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                    self.dUn.append(direction_index+1)
                elif boundary.var[i] == 'v':
                    self.nVn += 1
                    self.iVn.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                    self.dVn.append(direction_index+1)
                elif boundary.var[i] == 'w':
                    self.nWn += 1
                    self.iWn.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                    self.dWn.append(direction_index+1)
                elif boundary.var[i] == 'p':
                    self.nPn += 1
                    self.iPn.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                    self.dPn.append(direction_index+1)
                elif boundary.var[i] == 'e':
                    self.nEn += 1
                    self.iEn.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                    self.dEn.append(direction_index+1)
            elif boundary.type[i] == 'sec':
                direction_index = direction_order.index(boundary.dir[i])
                if boundary.var[i] == 'u':
                    self.nUs += 1
                    self.iUs.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                    self.dUs.append(direction_index+1)
                elif boundary.var[i] == 'v':
                    self.nVs += 1
                    self.iVs.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                    self.dVs.append(direction_index+1)
                elif boundary.var[i] == 'w':
                    self.nWs += 1
                    self.iWs.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                    self.dWs.append(direction_index+1)
                elif boundary.var[i] == 'p':
                    self.nPs += 1
                    self.iPs.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                    self.dPs.append(direction_index+1)
                elif boundary.var[i] == 'e':
                    self.nEs += 1
                    self.iEs.append([boundary.xi[i], boundary.xf[i], boundary.yi[i], boundary.yf[i], boundary.zi[i], boundary.zf[i]])
                    self.dEs.append(direction_index+1)

        # Configurações dos corners
        self.cL = boundary.corners.limits
        self.cD = boundary.corners.dir
        self.adiabatic = boundary.corners.adiabatic
        self.cN = len(self.cL)

        # Coeficientes de Neumann
        neumann_length = len(boundary.neumann_coeffs)
        neumann2_length = len(boundary.neumann2_coeffs)

        # Valor de gamma-1 para conversão de pressão em densidade
        self.gamma1 = boundary.gamma - 1

        # Configurações do campo elétrico
        self.E0 = boundary.E0

        # Divisão para diferentes processadores
        self.bi = [None] * (p_row * p_col)
        for j in range(p_row):
            for k in range(p_col):
                n_proc = k + j * p_col
                biL = deepcopy(self)  # Cria cópia temporária para este processador

                Ji = domainSlicesY[0, j]
                Jf = domainSlicesY[1, j]
                Ki = domainSlicesZ[0, k]
                Kf = domainSlicesZ[1, k]

                # Limitar índices (função externa)
                biL.iUd, biL.nUd, biL.vUd = self.limit_indices(biL.iUd, biL.nUd, biL.vUd, 'd', Ji, Jf, Ki, Kf, 0)
                biL.iVd, biL.nVd, biL.vVd = self.limit_indices(biL.iVd, biL.nVd, biL.vVd, 'd', Ji, Jf, Ki, Kf, 0)
                biL.iWd, biL.nWd, biL.vWd = self.limit_indices(biL.iWd, biL.nWd, biL.vWd, 'd', Ji, Jf, Ki, Kf, 0)
                biL.iPd, biL.nPd, biL.vPd = self.limit_indices(biL.iPd, biL.nPd, biL.vPd, 'd', Ji, Jf, Ki, Kf, 0)
                biL.iEd, biL.nEd, biL.vEd = self.limit_indices(biL.iEd, biL.nEd, biL.vEd, 'd', Ji, Jf, Ki, Kf, 0)

                biL.iUn, biL.nUn, biL.dUn = self.limit_indices(biL.iUn, biL.nUn, biL.dUn, 'n', Ji, Jf, Ki, Kf, neumann_length)
                biL.iVn, biL.nVn, biL.dVn = self.limit_indices(biL.iVn, biL.nVn, biL.dVn, 'n', Ji, Jf, Ki, Kf, neumann_length)
                biL.iWn, biL.nWn, biL.dWn = self.limit_indices(biL.iWn, biL.nWn, biL.dWn, 'n', Ji, Jf, Ki, Kf, neumann_length)
                biL.iPn, biL.nPn, biL.dPn = self.limit_indices(biL.iPn, biL.nPn, biL.dPn, 'n', Ji, Jf, Ki, Kf, neumann_length)
                biL.iEn, biL.nEn, biL.dEn = self.limit_indices(biL.iEn, biL.nEn, biL.dEn, 'n', Ji, Jf, Ki, Kf, neumann_length)

                biL.iUs, biL.nUs, biL.dUs = self.limit_indices(biL.iUs, biL.nUs, biL.dUs,'n',Ji,Jf,Ki,Kf,neumann2_length)
                biL.iVs, biL.nVs, biL.dVs = self.limit_indices(biL.iVs, biL.nVs, biL.dVs,'n',Ji,Jf,Ki,Kf,neumann2_length)
                biL.iWs, biL.nWs, biL.dWs = self.limit_indices(biL.iWs, biL.nWs, biL.dWs,'n',Ji,Jf,Ki,Kf,neumann2_length)
                biL.iPs, biL.nPs, biL.dPs = self.limit_indices(biL.iPs, biL.nPs, biL.dPs,'n',Ji,Jf,Ki,Kf,neumann2_length)
                biL.iEs, biL.nEs, biL.dEs = self.limit_indices(biL.iEs, biL.nEs, biL.dEs,'n',Ji,Jf,Ki,Kf,neumann2_length)
                # Similar para as outras variáveis...

                #values = [biL.cD, biL.adiabatic]
                values = np.hstack((np.array(biL.cD), biL.adiabatic))
                biL.cL, biL.cN, values = self.limit_indices(biL.cL, biL.cN, values, 'c', Ji, Jf, Ki, Kf, neumann_length)
                biL.cD = values[:,0:3]
                biL.adiabatic = values[:, 3]

                # Armazenar resultados no processador
                self.bi[n_proc] = biL
                
        self.split_disturbances(boundary, mesh, domainSlicesY, domainSlicesZ, p_row, p_col)
        
    
    def limit_indices(self, ind, n, vd, boundary_type, Ji, Jf, Ki, Kf, neumann_length):
        ind = np.array(ind)
        vd = np.array(vd)
        if n == 0:
            return ind, n, vd
        
        # Ajustando índices com max/min conforme o Matlab
        for i in range(n):
            ind[i,2:6] = [max(ind[i,2], Ji), min(ind[i,3], Jf), max(ind[i,4], Ki), min(ind[i,5], Kf)]
            #ind[i,2:6] = [max(ind[i,2], Ji), min(ind[i,3], Jf), max(ind[i,4], Ki), min(ind[i,5], Kf)]
        
        # Identificando índices a serem removidos
        to_remove = np.logical_or(ind[:, 2] > ind[:, 3], ind[:, 4] > ind[:, 5])
        
        # Removendo entradas inválidas
        ind = ind[~to_remove, :]

        # Atualizando 'vd' (vetor ou matriz) dependendo do tamanho
        if np.ndim(vd) == 1 and n > 1:
        # Caso vd seja um vetor (1D array) e n > 1, remover elementos com base em toRemove
            vd = np.delete(vd, np.where(to_remove)[0])  # np.where(toRemove) retorna os índices de True
        elif np.ndim(vd) == 1 and n == 1:
            vd = np.delete(vd, np.where(to_remove)[0])
        elif vd.size == 0:
            pass
        # Caso vd seja uma matriz (2D array), remover as linhas onde toRemove é True
            vd = vd[~to_remove, :]

        
        n -= sum(to_remove)
        
        # Tratamento para Neumann
        if boundary_type == 'n':
            for i in range(n):
                if vd[i] == 3:
                    if ind[i,3] + neumann_length > Jf:
                        raise ValueError(f"There is a y+ Neumann condition at J = {ind[i,3]} crossing a domain slice at J = {Jf}. Consider changing p_row.")
                elif vd[i] == 4:
                    if ind[i,2] - neumann_length < Ji:
                        raise ValueError(f"There is a y- Neumann condition at J = {ind[i,2]} crossing a domain slice at J = {Ji}. Consider changing p_row.")
                elif vd[i] == 5:
                    if ind[i,5] + neumann_length > Kf:
                        raise ValueError(f"There is a z+ Neumann condition at K = {ind[i,5]} crossing a domain slice at K = {Kf}. Consider changing p_col.")
                elif vd[i] == 6:
                    if ind[i,4] - neumann_length < Ki:
                        raise ValueError(f"There is a z- Neumann condition at K = {ind[i,4]} crossing a domain slice at K = {Ki}. Consider changing p_col.")
        
        return ind, n, vd

    def split_disturbances(self, boundary, mesh, domain_slices_y, domain_slices_z, p_row, p_col):
    
        # Iteração sobre processadores (j e k)
        for j in range(p_row):
            for k in range(p_col):
                n_proc = k + j * p_col  # Calcula o índice do processador atual (Python já está com indexação ajustada)
                disturb = []  # Lista que armazenará os distúrbios para o processador atual
                
                # Itera sobre todos os distúrbios em boundary.disturb
                for i in range(len(boundary.disturb)):
                    if boundary.disturb[i] is not None:
                        ind = boundary.disturb[i].ind
                        ind[2:6] = [max(ind[2], domain_slices_y[0][j]), min(ind[3], domain_slices_y[1][j]), 
                                    max(ind[4], domain_slices_z[0][k]), min(ind[5], domain_slices_z[1][k])]

                        # Verifica se o índice é válido
                        if ind[2] <= ind[3] and ind[4] <= ind[5]:
                            # Cria um novo objeto Disturbance para adicionar à lista disturb
                            new_disturb = Disturbance()
                            new_disturb.ind = ind
                            new_disturb.X = mesh.X[ind[0]:ind[1]+1]  # Ajusta o slice para Python
                            new_disturb.Y = mesh.Y[ind[2]:ind[3]+1]
                            new_disturb.Z = mesh.Z[ind[4]:ind[5]+1]
                            
                            disturb.append(new_disturb)
                
                # Atribui a lista de distúrbios ao processador correspondente
                self.bi[n_proc].disturb = disturb



class Wall:
    def __init__(self, wallUpLimits, wallDownLimits, wallFrontLimits, wallBackLimits, wallRightLimits, wallLeftLimits):
        self.up = wallUpLimits
        self.down = wallDownLimits
        self.front = wallFrontLimits
        self.back = wallBackLimits
        self.right = wallRightLimits
        self.left = wallLeftLimits

class Corners:
    def __init__(self):
        self.limits = []
        self.dir = []
        self.adiabatic = []
    def add_corner(self, limit, direction, adiabatic=0):
        self.limits.append(limit)
        self.dir.append(direction)
        self.adiabatic.append(adiabatic)

class BoundaryConditions:
    def __init__(self, flow_type, mesh, flow_parameters, neumann_order):
        self.gamma = flow_parameters.gamma
        self.E0 = 1 / ((self.gamma ** 2 - self.gamma) * flow_parameters.Ma ** 2)
        self.P0 = (self.gamma - 1) * self.E0
        self.var = []        # Variable to which the condition is applied
        self.type = []       # Type of boundary ('dir' for Dirichlet or 'neu' for Neumann)
        self.dir = []        # Direction of the derivative for Neumann condition
        self.val = []        # Value of the variable (Dirichlet) or derivative (Neumann)
        self.xi = []         # Starting j index for the condition
        self.xf = []         # Ending j index for the condition
        self.yi = []         # Starting i index for the condition
        self.yf = []         # Ending i index for the condition
        self.zi = []         # Starting k index for the condition
        self.zf = []         # Ending k index for the condition
        self.flow_region = None
        self.wall = {
            "up": None,
            "down": None,
            "front": None,
            "back": None,
            "right": None,
            "left": None
        }
        self.corners = None
        self.inside_wall = None
        self.neumann_coeffs = None
        self.neumann2_coeffs = None
        self.disturb = []
        self.wall = None

        self.mesh = mesh
        self.flow_type = flow_type
        self.flow_parameters = flow_parameters

        # Call subroutine based on flowType
        self.get_boundary_conditions(self.flow_type, neumann_order)
        self.add_disturbances()

        

    def get_boundary_conditions2(self, flow_type, neumann_order):
        file_path = f'source/boundaries/{flow_type.name}.py'
        if os.path.exists(file_path):
            # Dynamically load and execute boundary conditions script for specific flow_type
            exec(open(file_path).read())
        else:
            raise FileNotFoundError(f"Boundary condition file '{file_path}' not found.")

        self.inside_wall = np.logical_not(self.flow_region)
        walls = [self.wall["up"], self.wall["down"], self.wall["front"], self.wall["back"], self.wall["right"], self.wall["left"]]

        for i, current_wall in enumerate(walls):
            for j in range(current_wall.shape[0]):
                self.inside_wall[current_wall[j, 0]:current_wall[j, 1], current_wall[j, 2]:current_wall[j, 3], current_wall[j, 4]:current_wall[j, 5]] = False

        # Neumann condition coefficients
        self.get_neumann_coeffs(neumann_order)

    def get_boundary_conditions(self, flow_type, neumann_order):
        match flow_type.name:
            case 'boundaryLayerAdiabatic':
                self.__boundaryLayerAdiabatic()
            case 'boundaryLayerFreeSlip':
                self.__boundaryLayerFreeSlip()
            case 'boundaryLayerIsothermal':
                self.__boundaryLayerIsothermal()
            case 'boundaryLayerIsothermalPressureInlet':
                self.__boundaryLayerIsothermalPressureInlet()
            case 'boundaryLayerIsothermalSymmZ':
                self.__boundaryLayerIsothermalSymmZ()
            case 'lidDrivenFlow':
                self.__lidDrivenFlow()
            case 'periodicBox':
                self.__periodicBox()
            case 'poiseuilleFlow':
                self.__poiseuilleFlow()
            case _:
                print('Boundary condition not available. Check the spelling of the type of BL.')
            
        for i in range(0,6):
            match i:
                case 0:
                    currentWall = self.wall.up
                case 1:
                    currentWall = self.wall.down
                case 2:
                    currentWall = self.wall.front
                case 3:
                    currentWall = self.wall.back
                case 4:
                    currentWall = self.wall.right
                case 5:
                    currentWall = self.wall.left
            for j in range(0, currentWall.shape[0]):
                #self.inside_wall[currentWall[j,0]:currentWall[j,1]+1, currentWall[j,2]:currentWall[j,3]+1, currentWall[j,4]:currentWall[j,5]+1] = False
                self.inside_wall[currentWall[j,4]:currentWall[j,5]+1, currentWall[j,2]:currentWall[j,3]+1, currentWall[j,0]:currentWall[j,1]+1] = False
        
        # Neumann condition coefficients
        self.get_neumann_coeffs(neumann_order)   

    def __boundaryLayerAdiabatic(self):
        if self.mesh.X[0] <= 0:
            #u
            self.var.append('u')
            self.type.append('dir')
            self.dir.append('xi')
            self.val.append(1)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

            #v
            self.var.append('v')
            self.type.append('dir')
            self.dir.append('xi')
            self.val.append(0)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz) 

            # w
            self.var.append('w')
            self.type.append('dir')
            self.dir.append('xi')
            self.val.append(0)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

            # e
            self.var.append('e')
            self.type.append('dir')
            self.dir.append('xi')
            self.val.append(self.E0)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)
        else:
            if not hasattr(self.flow_type, 'disturb'):
                self.flow_type.disturb = []
            else:
                self.flow_type.disturb.append([])
                self.flow_type.disturb[1:-1] = self.flow_type.disturb[0]
            self.flow_type.disturb[0] = []
            self.flow_type.disturb[0].x = [self.mesh.X(1), self.mesh.X(1)]
            self.flow_type.disturb[0].y = [-np.inf, np.inf] #Verificar o funcionamento dos inf no python
            self.flow_type.disturb[0].z = [-np.inf, np.inf]
            self.flow_type.disturb[0].var = 'UVRWE'
            self.flow_type.disturb[0].type = 'holdInlet'
            self.flow_type.disturb[0].active = True
            self.flow_type.disturb[0].par = [0] #Hold density

        #p
        self.var.append('p')
        self.type.append('neu')
        self.dir.append('xi')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(1)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)
    
        ## Outflow
        # u
        self.var.append('u')
        self.type.append('sec')
        self.dir.append('xf')
        self.val.append(0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # v
        self.var.append('v')
        self.type.append('sec')
        self.dir.append('xf')
        self.val.append(0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # w
        self.var.append('w')
        self.type.append('sec')
        self.dir.append('xf')
        self.val.append(0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # e
        self.var.append('e')
        self.type.append('sec')
        self.dir.append('xf')
        self.val.append(0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # p
        self.var.append('p')
        self.type.append('dir')
        self.dir.append('xf')
        self.val.append(self.P0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        ## Outerflow
        # u
        self.var.append('u')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # v
        self.var.append('v')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # w
        self.var.append('w')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # e
        self.var.append('e')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # p
        self.var.append('p')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        ## Outerflow for non-periodic 3D
        if self.mesh.nz > 1 and not self.mesh.z.periodic:

            # u
            self.var.append('u') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # v
            self.var.append('v') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # w
            self.var.append('w') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # e
            self.var.append('e') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # p
            self.var.append('p') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 
            
            # u
            self.var.append('u') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # v
            self.var.append('v') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # w
            self.var.append('w') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # e
            self.var.append('e') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # p
            self.var.append('p') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

        ## Define flow region
        # Find which nodes will actually contain a flow and which ones will be in or at a wall
        flowRegion = np.ones((self.mesh.nz, self.mesh.ny, self.mesh.nx)) #essa função tem um comportamento diferente da função TRUE do matlab
        # Add flat plate
        wallJ = np.argmin(np.abs(self.mesh.Y))
        flowRegion[:, 0:wallJ+1, :] = False

        # Add cavities to the flow region
        if hasattr(self.flow_type, 'cav'):
            for i in range(len(self.flow_type.cav)):
                x = self.flow_type.cav[i].x
                y = self.flow_type.cav[i].y
                z = self.flow_type.cav[i].z

                mask_x = (self.mesh.X > x[0]) & (self.mesh.X < x[1])
                mask_y = (self.mesh.Y > y[0]) & (self.mesh.Y < y[1])
                mask_z = (self.mesh.Z > z[0]) & (self.mesh.Z < z[1])

                # Usar numpy.ix_ para aplicar as máscaras booleanas em múltiplas dimensões
                flowRegion[np.ix_(mask_z, mask_y, mask_x)] = True

        # Remove roughnesses from the flow
        if hasattr(self.flow_type, 'rug'):
            for i in range(len(self.flow_type.rug)):
                x = self.flow_type.rug[i].x
                y = self.flow_type.rug[i].y
                z = self.flow_type.rug[i].z

                mask_x = (self.mesh.X >= x[0]) & (self.mesh.X <= x[1])
                mask_y = (self.mesh.Y >= y[0]) & (self.mesh.Y <= y[1])
                mask_z = (self.mesh.Z >= z[0]) & (self.mesh.Z <= z[1])

                flowRegion[np.ix_(mask_z, mask_y, mask_x)] = False
        self.flow_region = flowRegion
        [wallFrontLimits, wallBackLimits, wallUpLimits, wallDownLimits, wallRightLimits, wallLeftLimits, insideWalls, corners] = self.__findWallsForBoundaries(flowRegion)

        corners.adiabatic = np.ones((len(corners.dir),1))
        ## Create wall for free-slip region
        if self.mesh.X[0] < 0:
            wallUpLimits[:, 0] = np.maximum(wallUpLimits[:, 0], np.argmax(self.mesh.X >= 0))
            wallUpLimits = np.vstack([wallUpLimits, [0, np.where(self.mesh.X < 0)[0][-1], 
                                             np.argmax(self.mesh.Y >= 0), 
                                             np.argmax(self.mesh.Y >= 0), 
                                             0, self.mesh.nz - 1]])

            self.mesh.x.breakPoint = np.array([np.where(self.mesh.X < 0)[0][-1], 
                              np.argmax(self.mesh.Y >= 0), 
                              np.argmax(self.mesh.Y >= 0), 
                              0, self.mesh.nz - 1])

        for i in range(1,7):
            match i:
                case 1:
                    wallPosition = wallFrontLimits
                    wallDir = 'xi'
                case 2:
                    wallPosition = wallBackLimits
                    wallDir = 'xf'  
                case 3:
                    wallPosition = wallUpLimits
                    wallDir = 'yi'
                case 4:
                    wallPosition = wallDownLimits
                    wallDir = 'yf'
                case 5:
                    wallPosition = wallLeftLimits
                    wallDir = 'zi'
                case 6:
                    wallPosition = wallRightLimits
                    wallDir = 'zf'
        
            for j in range(0, wallPosition.shape[0]):
                self.var.append('p')  #%#ok<*SAGROW>
                self.type.append('neu')
                self.dir.append(wallDir)
                self.val.append(0)
                self.xi.append(wallPosition[j,0]+1)
                self.xf.append(wallPosition[j,1]+1)
                self.yi.append(wallPosition[j,2]+1)
                self.yf.append(wallPosition[j,3]+1)
                self.zi.append(wallPosition[j,4]+1)
                self.zf.append(wallPosition[j,5]+1)


                self.var.append('e')  #%#ok<*SAGROW>
                self.type.append('neu')
                self.dir.append(wallDir)
                self.val.append(0)
                self.xi.append(wallPosition[j,0]+1)
                self.xf.append(wallPosition[j,1]+1)
                self.yi.append(wallPosition[j,2]+1)
                self.yf.append(wallPosition[j,3]+1)
                self.zi.append(wallPosition[j,4]+1)
                self.zf.append(wallPosition[j,5]+1)
        
        ## Add free slip wall, it is the last up facing wall
        if self.mesh.X[0] < 0:
            self.var.append('u')
            self.type.append('neu')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(wallUpLimits[-1,0]+1)
            self.xf.append(wallUpLimits[-1,1]+1)
            self.yi.append(wallUpLimits[-1,2]+1)
            self.yf.append(wallUpLimits[-1,3]+1)
            self.zi.append(wallUpLimits[-1,4]+1)
            self.zf.append(wallUpLimits[-1,5]+1)

            self.var.append('w')
            self.type.append('neu')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(wallUpLimits[-1,0]+1)
            self.xf.append(wallUpLimits[-1,1]+1)
            self.yi.append(wallUpLimits[-1,2]+1)
            self.yf.append(wallUpLimits[-1,3]+1)
            self.zi.append(wallUpLimits[-1,4]+1)
            self.zf.append(wallUpLimits[-1,5]+1)
    
        ## Add regions that are inside walls
        for i in range(0, insideWalls.shape[0]):
            self.var.append('p')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(self.P0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)

            self.var.append('u')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            self.var.append('v')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            self.var.append('w')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            self.var.append('e')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(self.E0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
        
        self.corners = corners
        self.wall = Wall(wallUpLimits, wallDownLimits, wallFrontLimits, wallBackLimits, wallRightLimits, wallLeftLimits)
        self.inside_wall = np.logical_not(flowRegion)

    def __boundaryLayerFreeSlip(self):
        if not self.mesh.x.periodic:
            if self.mesh.X[0] <= 0:
                #u
                self.var.append('u')
                self.type.append('dir')
                self.dir.append('xi')
                self.val.append(1)
                self.xi.append(1)
                self.xf.append(1)
                self.yi.append(1)
                self.yf.append(self.mesh.ny)
                self.zi.append(1)
                self.zf.append(self.mesh.nz)

                #v
                self.var.append('v')
                self.type.append('dir')
                self.dir.append('xi')
                self.val.append(0)
                self.xi.append(1)
                self.xf.append(1)
                self.yi.append(1)
                self.yf.append(self.mesh.ny)
                self.zi.append(1)
                self.zf.append(self.mesh.nz) 

                # w
                self.var.append('w')
                self.type.append('dir')
                self.dir.append('xi')
                self.val.append(0)
                self.xi.append(1)
                self.xf.append(1)
                self.yi.append(1)
                self.yf.append(self.mesh.ny)
                self.zi.append(1)
                self.zf.append(self.mesh.nz)

                # e
                self.var.append('e')
                self.type.append('dir')
                self.dir.append('xi')
                self.val.append(self.E0)
                self.xi.append(1)
                self.xf.append(1)
                self.yi.append(1)
                self.yf.append(self.mesh.ny)
                self.zi.append(1)
                self.zf.append(self.mesh.nz)
            else:
                if not hasattr(self.flow_type, 'disturb'):
                    self.flow_type.disturb = []
                else:
                    self.flow_type.disturb.append([])
                    self.flow_type.disturb[1:-1] = self.flow_type.disturb[0]
                self.flow_type.disturb[0] = []
                self.flow_type.disturb[0].x = [self.mesh.X(1), self.mesh.X(1)]
                self.flow_type.disturb[0].y = [-np.inf, np.inf] #Verificar o funcionamento dos inf no python
                self.flow_type.disturb[0].z = [-np.inf, np.inf]
                self.flow_type.disturb[0].var = 'UVRWE'
                self.flow_type.disturb[0].type = 'holdInlet'
                self.flow_type.disturb[0].active = True
                self.flow_type.disturb[0].par = [0] #Hold density

            #p
            self.var.append('p')
            self.type.append('neu')
            self.dir.append('xi')
            self.val.append(0)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)
        
            ## Outflow
            # u
            self.var.append('u')
            self.type.append('sec')
            self.dir.append('xf')
            self.val.append(0)
            self.xi.append(self.mesh.nx)
            self.xf.append(self.mesh.nx)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

            # v
            self.var.append('v')
            self.type.append('sec')
            self.dir.append('xf')
            self.val.append(0)
            self.xi.append(self.mesh.nx)
            self.xf.append(self.mesh.nx)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

            # w
            self.var.append('w')
            self.type.append('sec')
            self.dir.append('xf')
            self.val.append(0)
            self.xi.append(self.mesh.nx)
            self.xf.append(self.mesh.nx)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

            # e
            self.var.append('e')
            self.type.append('sec')
            self.dir.append('xf')
            self.val.append(0)
            self.xi.append(self.mesh.nx)
            self.xf.append(self.mesh.nx)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

            # p
            self.var.append('p')
            self.type.append('dir')
            self.dir.append('xf')
            self.val.append(self.P0)
            self.xi.append(self.mesh.nx)
            self.xf.append(self.mesh.nx)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

        if not self.mesh.y.periodic:
            ## Outerflow
            # u
            self.var.append('u')
            self.type.append('sec')
            self.dir.append('yf')
            self.val.append(0)
            self.xi.append(1)
            self.xf.append(self.mesh.nx)
            self.yi.append(self.mesh.ny)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

            # v
            self.var.append('v')
            self.type.append('sec')
            self.dir.append('yf')
            self.val.append(0)
            self.xi.append(1)
            self.xf.append(self.mesh.nx)
            self.yi.append(self.mesh.ny)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

            # w
            self.var.append('w')
            self.type.append('sec')
            self.dir.append('yf')
            self.val.append(0)
            self.xi.append(1)
            self.xf.append(self.mesh.nx)
            self.yi.append(self.mesh.ny)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

            # e
            self.var.append('e')
            self.type.append('sec')
            self.dir.append('yf')
            self.val.append(0)
            self.xi.append(1)
            self.xf.append(self.mesh.nx)
            self.yi.append(self.mesh.ny)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

            # p
            self.var.append('p')
            self.type.append('sec')
            self.dir.append('yf')
            self.val.append(0)
            self.xi.append(1)
            self.xf.append(self.mesh.nx)
            self.yi.append(self.mesh.ny)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

        ## Outerflow for non-periodic 3D
        if self.mesh.nz > 1 and not self.mesh.z.periodic:

            # u
            self.var.append('u') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # v
            self.var.append('v') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # w
            self.var.append('w') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # e
            self.var.append('e') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # p
            self.var.append('p') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 
            
            # u
            self.var.append('u') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # v
            self.var.append('v') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # w
            self.var.append('w') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # e
            self.var.append('e') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # p
            self.var.append('p') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

        ## Define flow region
        # Find which nodes will actually contain a flow and which ones will be in or at a wall
        flowRegion = np.ones((self.mesh.nz, self.mesh.ny, self.mesh.nx)) #essa função tem um comportamento diferente da função TRUE do matlab
        if not self.mesh.y.periodic:
            # Add flat plate
            wallJ = np.argmin(np.abs(self.mesh.Y))
            flowRegion[:, 0:wallJ+1, :] = False

        # Add cavities to the flow region
        if hasattr(self.flow_type, 'cav'):
            for i in range(len(self.flow_type.cav)):
                x = self.flow_type.cav[i].x
                y = self.flow_type.cav[i].y
                z = self.flow_type.cav[i].z

                mask_x = (self.mesh.X > x[0]) & (self.mesh.X < x[1])
                mask_y = (self.mesh.Y > y[0]) & (self.mesh.Y < y[1])
                mask_z = (self.mesh.Z > z[0]) & (self.mesh.Z < z[1])

                # Usar numpy.ix_ para aplicar as máscaras booleanas em múltiplas dimensões
                flowRegion[np.ix_(mask_z, mask_y, mask_x)] = True

        # Remove roughnesses from the flow
        if hasattr(self.flow_type, 'rug'):
            for i in range(len(self.flow_type.rug)):
                x = self.flow_type.rug[i].x
                y = self.flow_type.rug[i].y
                z = self.flow_type.rug[i].z

                mask_x = (self.mesh.X >= x[0]) & (self.mesh.X <= x[1])
                mask_y = (self.mesh.Y >= y[0]) & (self.mesh.Y <= y[1])
                mask_z = (self.mesh.Z >= z[0]) & (self.mesh.Z <= z[1])

                flowRegion[np.ix_(mask_z, mask_y, mask_x)] = False
        self.flow_region = flowRegion
        [wallFrontLimits, wallBackLimits, wallUpLimits, wallDownLimits, wallRightLimits, wallLeftLimits, insideWalls, corners] = self.__findWallsForBoundaries(flowRegion)


        for i in range(1,7):
            match i:
                case 1:
                    wallPosition = wallFrontLimits
                    wallDir = 'xi'
                case 2:
                    wallPosition = wallBackLimits
                    wallDir = 'xf'  
                case 3:
                    wallPosition = wallUpLimits
                    wallDir = 'yi'
                case 4:
                    wallPosition = wallDownLimits
                    wallDir = 'yf'
                case 5:
                    wallPosition = wallRightLimits
                    wallDir = 'zi'
                case 6:
                    wallPosition = wallLeftLimits
                    wallDir = 'zf'
        
            for j in range(0, wallPosition.shape[0]):
                self.var.append('u')  #%#ok<*SAGROW>
                self.type.append('neu')
                self.dir.append(wallDir)
                self.val.append(0)
                self.xi.append(wallPosition[j,0]+1)
                self.xf.append(wallPosition[j,1]+1)
                self.yi.append(wallPosition[j,2]+1)
                self.yf.append(wallPosition[j,3]+1)
                self.zi.append(wallPosition[j,4]+1)
                self.zf.append(wallPosition[j,5]+1)

            for j in range(0, wallPosition.shape[0]):
                self.var.append('w')
                self.type.append('neu')
                self.dir.append(wallDir)
                self.val.append(0)
                self.xi.append(wallPosition[j,0]+1)
                self.xf.append(wallPosition[j,1]+1)
                self.yi.append(wallPosition[j,2]+1)
                self.yf.append(wallPosition[j,3]+1)
                self.zi.append(wallPosition[j,4]+1)
                self.zf.append(wallPosition[j,5]+1)

            for j in range(0, wallPosition.shape[0]):
                self.var.append('p')  #%#ok<*SAGROW>
                self.type.append('neu')
                self.dir.append(wallDir)
                self.val.append(0)
                self.xi.append(wallPosition[j,0]+1)
                self.xf.append(wallPosition[j,1]+1)
                self.yi.append(wallPosition[j,2]+1)
                self.yf.append(wallPosition[j,3]+1)
                self.zi.append(wallPosition[j,4]+1)
                self.zf.append(wallPosition[j,5]+1)

    
        ## Add regions that are inside walls
        for i in range(0, insideWalls.shape[0]):
            self.var.append('p')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(self.P0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)

            self.var.append('u')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            self.var.append('v')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            self.var.append('w')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)

            if hasattr(self.flow_type, 'tWallRelative'):
                eWall = self.E0 * self.flow_type.tWallRelative
            else:
                eWall = self.E0

            self.var.append('e')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(eWall)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
        
        self.corners = corners
        self.wall = Wall(wallUpLimits, wallDownLimits, wallFrontLimits, wallBackLimits, wallRightLimits, wallLeftLimits)
        self.inside_wall = np.logical_not(flowRegion)
    
    def __boundaryLayerIsothermal(self):
        if self.mesh.X[0] <= 0:
            #u
            self.var.append('u')
            self.type.append('dir')
            self.dir.append('xi')
            self.val.append(1)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

            #v
            self.var.append('v')
            self.type.append('dir')
            self.dir.append('xi')
            self.val.append(0)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz) 

            # w
            self.var.append('w')
            self.type.append('dir')
            self.dir.append('xi')
            self.val.append(0)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

            # e
            self.var.append('e')
            self.type.append('dir')
            self.dir.append('xi')
            self.val.append(self.E0)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)
        else:
            if not hasattr(self.flow_type, 'disturb'):
                self.flow_type.disturb = []
            else:
                self.flow_type.disturb.append([])
                self.flow_type.disturb[1:-1] = self.flow_type.disturb[0]
            self.flow_type.disturb[0] = []
            self.flow_type.disturb[0].x = [self.mesh.X(1), self.mesh.X(1)]
            self.flow_type.disturb[0].y = [-np.inf, np.inf] #Verificar o funcionamento dos inf no python
            self.flow_type.disturb[0].z = [-np.inf, np.inf]
            self.flow_type.disturb[0].var = 'UVRWE'
            self.flow_type.disturb[0].type = 'holdInlet'
            self.flow_type.disturb[0].active = True
            self.flow_type.disturb[0].par = [0] #Hold density

        #p
        self.var.append('p')
        self.type.append('neu')
        self.dir.append('xi')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(1)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)
    
        ## Outflow
        # u
        self.var.append('u')
        self.type.append('sec')
        self.dir.append('xf')
        self.val.append(0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # v
        self.var.append('v')
        self.type.append('sec')
        self.dir.append('xf')
        self.val.append(0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # w
        self.var.append('w')
        self.type.append('sec')
        self.dir.append('xf')
        self.val.append(0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # e
        self.var.append('e')
        self.type.append('sec')
        self.dir.append('xf')
        self.val.append(0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # p
        self.var.append('p')
        self.type.append('dir')
        self.dir.append('xf')
        self.val.append(self.P0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        ## Outerflow
        # u
        self.var.append('u')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # v
        self.var.append('v')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # w
        self.var.append('w')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # e
        self.var.append('e')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # p
        self.var.append('p')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        ## Outerflow for non-periodic 3D
        if self.mesh.nz > 1 and not self.mesh.z.periodic:

            # u
            self.var.append('u') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # v
            self.var.append('v') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # w
            self.var.append('w') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # e
            self.var.append('e') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # p
            self.var.append('p') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 
            
            # u
            self.var.append('u') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # v
            self.var.append('v') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # w
            self.var.append('w') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # e
            self.var.append('e') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # p
            self.var.append('p') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

        ## Define flow region
        # Find which nodes will actually contain a flow and which ones will be in or at a wall
        flowRegion = np.ones((self.mesh.nz, self.mesh.ny, self.mesh.nx)) #essa função tem um comportamento diferente da função TRUE do matlab
        # Add flat plate
        wallJ = np.argmin(np.abs(self.mesh.Y))
        flowRegion[:, 0:wallJ+1, :] = False

        # Add cavities to the flow region
        if hasattr(self.flow_type, 'cav'):
            for i in range(len(self.flow_type.cav)):
                x = self.flow_type.cav[i].x
                y = self.flow_type.cav[i].y
                z = self.flow_type.cav[i].z

                mask_x = (self.mesh.X > x[0]) & (self.mesh.X < x[1])
                mask_y = (self.mesh.Y > y[0]) & (self.mesh.Y < y[1])
                mask_z = (self.mesh.Z > z[0]) & (self.mesh.Z < z[1])

                # Usar numpy.ix_ para aplicar as máscaras booleanas em múltiplas dimensões
                flowRegion[np.ix_(mask_z, mask_y, mask_x)] = True

        # Remove roughnesses from the flow
        if hasattr(self.flow_type, 'rug'):
            for i in range(len(self.flow_type.rug)):
                x = self.flow_type.rug[i].x
                y = self.flow_type.rug[i].y
                z = self.flow_type.rug[i].z

                mask_x = (self.mesh.X >= x[0]) & (self.mesh.X <= x[1])
                mask_y = (self.mesh.Y >= y[0]) & (self.mesh.Y <= y[1])
                mask_z = (self.mesh.Z >= z[0]) & (self.mesh.Z <= z[1])

                flowRegion[np.ix_(mask_z, mask_y, mask_x)] = False
        self.flow_region = flowRegion
        [wallFrontLimits, wallBackLimits, wallUpLimits, wallDownLimits, wallRightLimits, wallLeftLimits, insideWalls, corners] = self.__findWallsForBoundaries(flowRegion)

        ## Create wall for free-slip region
        if self.mesh.X[0] < 0:
            wallUpLimits[:, 0] = np.maximum(wallUpLimits[:, 0], np.argmax(self.mesh.X >= 0))
            wallUpLimits = np.vstack([wallUpLimits, [0, np.where(self.mesh.X < 0)[0][-1], 
                                             np.argmax(self.mesh.Y >= 0), 
                                             np.argmax(self.mesh.Y >= 0), 
                                             0, self.mesh.nz - 1]])

            self.mesh.x.breakPoint = np.array([np.where(self.mesh.X < 0)[0][-1], 
                              np.argmax(self.mesh.Y >= 0), 
                              np.argmax(self.mesh.Y >= 0), 
                              0, self.mesh.nz - 1])

        for i in range(1,7):
            match i:
                case 1:
                    wallPosition = wallFrontLimits
                    wallDir = 'xi'
                case 2:
                    wallPosition = wallBackLimits
                    wallDir = 'xf'  
                case 3:
                    wallPosition = wallUpLimits
                    wallDir = 'yi'
                case 4:
                    wallPosition = wallDownLimits
                    wallDir = 'yf'
                case 5:
                    wallPosition = wallRightLimits
                    wallDir = 'zi'
                case 6:
                    wallPosition = wallLeftLimits
                    wallDir = 'zf'
        
            for j in range(0, wallPosition.shape[0]):
                self.var.append('p')  #%#ok<*SAGROW>
                self.type.append('neu')
                self.dir.append(wallDir)
                self.val.append(0)
                self.xi.append(wallPosition[j,0]+1)
                self.xf.append(wallPosition[j,1]+1)
                self.yi.append(wallPosition[j,2]+1)
                self.yf.append(wallPosition[j,3]+1)
                self.zi.append(wallPosition[j,4]+1)
                self.zf.append(wallPosition[j,5]+1)

        
        ## Add free slip wall, it is the last up facing wall
        if self.mesh.X[0] < 0:
            self.var.append('u')
            self.type.append('neu')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(wallUpLimits[-1,0]+1)
            self.xf.append(wallUpLimits[-1,1]+1)
            self.yi.append(wallUpLimits[-1,2]+1)
            self.yf.append(wallUpLimits[-1,3]+1)
            self.zi.append(wallUpLimits[-1,4]+1)
            self.zf.append(wallUpLimits[-1,5]+1)

            self.var.append('w')
            self.type.append('neu')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(wallUpLimits[-1,0]+1)
            self.xf.append(wallUpLimits[-1,1]+1)
            self.yi.append(wallUpLimits[-1,2]+1)
            self.yf.append(wallUpLimits[-1,3]+1)
            self.zi.append(wallUpLimits[-1,4]+1)
            self.zf.append(wallUpLimits[-1,5]+1)
    
        ## Add regions that are inside walls
        for i in range(0, insideWalls.shape[0]):
            self.var.append('p')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(self.P0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)

            self.var.append('u')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            self.var.append('v')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            self.var.append('w')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)

            if hasattr(self.flow_type,'tWallRelative'):
                eWall = self.E0*self.flow_type.tWallRelative
            else:
                eWall = self.E0
            
            self.var.append('e')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(self.E0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
        
        self.corners = corners
        self.wall = Wall(wallUpLimits, wallDownLimits, wallFrontLimits, wallBackLimits, wallRightLimits, wallLeftLimits)
        self.inside_wall = np.logical_not(flowRegion)

    def __boundaryLayerIsothermalPressureInlet(self):
        if self.mesh.X[0] <= 0:
            #u
            self.var.append('u')
            self.type.append('dir')
            self.dir.append('xi')
            self.val.append(1)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

            #v
            self.var.append('v')
            self.type.append('dir')
            self.dir.append('xi')
            self.val.append(0)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz) 

            # w
            self.var.append('w')
            self.type.append('dir')
            self.dir.append('xi')
            self.val.append(0)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

            # e
            self.var.append('e')
            self.type.append('dir')
            self.dir.append('xi')
            self.val.append(self.E0)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

            # p
            self.var.append('p')
            self.type.append('dir')
            self.dir.append('xi')
            self.val.append(self.P0)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)
        else:
            if not hasattr(self.flow_type, 'disturb'):
                self.flow_type.disturb = []
            else:
                self.flow_type.disturb.append([])
                self.flow_type.disturb[1:-1] = self.flow_type.disturb[0]
            self.flow_type.disturb[0] = []
            self.flow_type.disturb[0].x = [self.mesh.X(1), self.mesh.X(1)]
            self.flow_type.disturb[0].y = [-np.inf, np.inf] #Verificar o funcionamento dos inf no python
            self.flow_type.disturb[0].z = [-np.inf, np.inf]
            self.flow_type.disturb[0].var = 'UVRWE'
            self.flow_type.disturb[0].type = 'holdInlet'
            self.flow_type.disturb[0].active = True
            self.flow_type.disturb[0].par = [1] #Hold density
  
        ## Outflow
        # u
        self.var.append('u')
        self.type.append('sec')
        self.dir.append('xf')
        self.val.append(0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # v
        self.var.append('v')
        self.type.append('sec')
        self.dir.append('xf')
        self.val.append(0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # w
        self.var.append('w')
        self.type.append('sec')
        self.dir.append('xf')
        self.val.append(0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # e
        self.var.append('e')
        self.type.append('sec')
        self.dir.append('xf')
        self.val.append(0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # p
        self.var.append('p')
        self.type.append('neu')
        self.dir.append('xf')
        self.val.append(0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        ## Outerflow
        # u
        self.var.append('u')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # v
        self.var.append('v')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # w
        self.var.append('w')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # e
        self.var.append('e')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # p
        self.var.append('p')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        ## Outerflow for non-periodic 3D
        if self.mesh.nz > 1 and not self.mesh.z.periodic:

            # u
            self.var.append('u') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # v
            self.var.append('v') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # w
            self.var.append('w') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # e
            self.var.append('e') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # p
            self.var.append('p') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 
            
            # u
            self.var.append('u') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # v
            self.var.append('v') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # w
            self.var.append('w') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # e
            self.var.append('e') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # p
            self.var.append('p') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

        ## Define flow region
        # Find which nodes will actually contain a flow and which ones will be in or at a wall
        flowRegion = np.ones((self.mesh.nz, self.mesh.ny, self.mesh.nx)) #essa função tem um comportamento diferente da função TRUE do matlab
        # Add flat plate
        wallJ = np.argmin(np.abs(self.mesh.Y))
        flowRegion[:, 0:wallJ+1, :] = False

        # Add cavities to the flow region
        if hasattr(self.flow_type, 'cav'):
            for i in range(len(self.flow_type.cav)):
                x = self.flow_type.cav[i].x
                y = self.flow_type.cav[i].y
                z = self.flow_type.cav[i].z

                mask_x = (self.mesh.X > x[0]) & (self.mesh.X < x[1])
                mask_y = (self.mesh.Y > y[0]) & (self.mesh.Y < y[1])
                mask_z = (self.mesh.Z > z[0]) & (self.mesh.Z < z[1])

                # Usar numpy.ix_ para aplicar as máscaras booleanas em múltiplas dimensões
                flowRegion[np.ix_(mask_z, mask_y, mask_x)] = True

        # Remove roughnesses from the flow
        if hasattr(self.flow_type, 'rug'):
            for i in range(len(self.flow_type.rug)):
                x = self.flow_type.rug[i].x
                y = self.flow_type.rug[i].y
                z = self.flow_type.rug[i].z

                mask_x = (self.mesh.X >= x[0]) & (self.mesh.X <= x[1])
                mask_y = (self.mesh.Y >= y[0]) & (self.mesh.Y <= y[1])
                mask_z = (self.mesh.Z >= z[0]) & (self.mesh.Z <= z[1])

                flowRegion[np.ix_(mask_z, mask_y, mask_x)] = False
        self.flow_region = flowRegion
        [wallFrontLimits, wallBackLimits, wallUpLimits, wallDownLimits, wallRightLimits, wallLeftLimits, insideWalls, corners] = self.__findWallsForBoundaries(flowRegion)

        ## Create wall for free-slip region
        if self.mesh.X[0] < 0:
            wallUpLimits[:, 0] = np.maximum(wallUpLimits[:, 0], np.argmax(self.mesh.X >= 0))
            wallUpLimits = np.vstack([wallUpLimits, [0, np.where(self.mesh.X < 0)[0][-1], 
                                             np.argmax(self.mesh.Y >= 0), 
                                             np.argmax(self.mesh.Y >= 0), 
                                             0, self.mesh.nz - 1]])

            self.mesh.x.breakPoint = np.array([np.where(self.mesh.X < 0)[0][-1], 
                              np.argmax(self.mesh.Y >= 0), 
                              np.argmax(self.mesh.Y >= 0), 
                              0, self.mesh.nz - 1])

        for i in range(1,7):
            match i:
                case 1:
                    wallPosition = wallFrontLimits
                    wallDir = 'xi'
                case 2:
                    wallPosition = wallBackLimits
                    wallDir = 'xf'  
                case 3:
                    wallPosition = wallUpLimits
                    wallDir = 'yi'
                case 4:
                    wallPosition = wallDownLimits
                    wallDir = 'yf'
                case 5:
                    wallPosition = wallRightLimits
                    wallDir = 'zi'
                case 6:
                    wallPosition = wallLeftLimits
                    wallDir = 'zf'
        
            for j in range(0, wallPosition.shape[0]):
                self.var.append('p')  #%#ok<*SAGROW>
                self.type.append('neu')
                self.dir.append(wallDir)
                self.val.append(0)
                self.xi.append(wallPosition[j,0]+1)
                self.xf.append(wallPosition[j,1]+1)
                self.yi.append(wallPosition[j,2]+1)
                self.yf.append(wallPosition[j,3]+1)
                self.zi.append(wallPosition[j,4]+1)
                self.zf.append(wallPosition[j,5]+1)
        
        ## Add free slip wall, it is the last up facing wall
        if self.mesh.X[0] < 0:
            self.var.append('u')
            self.type.append('neu')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(wallUpLimits[-1,0]+1)
            self.xf.append(wallUpLimits[-1,1]+1)
            self.yi.append(wallUpLimits[-1,2]+1)
            self.yf.append(wallUpLimits[-1,3]+1)
            self.zi.append(wallUpLimits[-1,4]+1)
            self.zf.append(wallUpLimits[-1,5]+1)

            self.var.append('w')
            self.type.append('neu')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(wallUpLimits[-1,0]+1)
            self.xf.append(wallUpLimits[-1,1]+1)
            self.yi.append(wallUpLimits[-1,2]+1)
            self.yf.append(wallUpLimits[-1,3]+1)
            self.zi.append(wallUpLimits[-1,4]+1)
            self.zf.append(wallUpLimits[-1,5]+1)
    
        ## Add regions that are inside walls
        for i in range(0, insideWalls.shape[0]):
            self.var.append('p')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(self.P0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)

            self.var.append('u')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            self.var.append('v')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            self.var.append('w')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)

            if hasattr(self.flow_type,'tWallRelative'):
                eWall = self.E0*self.flow_type.tWallRelative
            else:
                eWall = self.E0
            
            self.var.append('e')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(eWall)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
        
        self.corners = corners
        self.wall = Wall(wallUpLimits, wallDownLimits, wallFrontLimits, wallBackLimits, wallRightLimits, wallLeftLimits)
        self.inside_wall = np.logical_not(flowRegion)

    def __boundaryLayerIsothermalSymmZ(self):
        if self.mesh.X[0] <= 0:
            #u
            self.var.append('u')
            self.type.append('dir')
            self.dir.append('xi')
            self.val.append(1)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

            #v
            self.var.append('v')
            self.type.append('dir')
            self.dir.append('xi')
            self.val.append(0)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz) 

            # w
            self.var.append('w')
            self.type.append('dir')
            self.dir.append('xi')
            self.val.append(0)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)

            # e
            self.var.append('e')
            self.type.append('dir')
            self.dir.append('xi')
            self.val.append(self.E0)
            self.xi.append(1)
            self.xf.append(1)
            self.yi.append(1)
            self.yf.append(self.mesh.ny)
            self.zi.append(1)
            self.zf.append(self.mesh.nz)
        else:
            if not hasattr(self.flow_type, 'disturb'):
                self.flow_type.disturb = []
            else:
                self.flow_type.disturb.append([])
                self.flow_type.disturb[1:-1] = self.flow_type.disturb[0]
            self.flow_type.disturb[0] = []
            self.flow_type.disturb[0].x = [self.mesh.X(1), self.mesh.X(1)]
            self.flow_type.disturb[0].y = [-np.inf, np.inf] #Verificar o funcionamento dos inf no python
            self.flow_type.disturb[0].z = [-np.inf, np.inf]
            self.flow_type.disturb[0].var = 'UVRWE'
            self.flow_type.disturb[0].type = 'holdInlet'
            self.flow_type.disturb[0].active = True
            self.flow_type.disturb[0].par = [0] #Hold density

        #p
        self.var.append('p')
        self.type.append('neu')
        self.dir.append('xi')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(1)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)
    
        ## Outflow
        # u
        self.var.append('u')
        self.type.append('sec')
        self.dir.append('xf')
        self.val.append(0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # v
        self.var.append('v')
        self.type.append('sec')
        self.dir.append('xf')
        self.val.append(0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # w
        self.var.append('w')
        self.type.append('sec')
        self.dir.append('xf')
        self.val.append(0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # e
        self.var.append('e')
        self.type.append('sec')
        self.dir.append('xf')
        self.val.append(0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # p
        self.var.append('p')
        self.type.append('dir')
        self.dir.append('xf')
        self.val.append(self.P0)
        self.xi.append(self.mesh.nx)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        ## Outerflow
        # u
        self.var.append('u')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # v
        self.var.append('v')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # w
        self.var.append('w')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # e
        self.var.append('e')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        # p
        self.var.append('p')
        self.type.append('sec')
        self.dir.append('yf')
        self.val.append(0)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        ## Outerflow for symmetry
        if self.mesh.nz > 1:

            # u
            self.var.append('u') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # v
            self.var.append('v') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # w
            self.var.append('w') 
            self.type.append('dir') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # e
            self.var.append('e') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 

            # p
            self.var.append('p') 
            self.type.append('neu') 
            self.dir.append('zi') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(1) 
            self.zf.append(1) 
            
            # u
            self.var.append('u') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # v
            self.var.append('v') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # w
            self.var.append('w') 
            self.type.append('dir') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # e
            self.var.append('e') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

            # p
            self.var.append('p') 
            self.type.append('neu') 
            self.dir.append('zf') 
            self.val.append(0) 
            self.xi.append(1) 
            self.xf.append(self.mesh.nx) 
            self.yi.append(1) 
            self.yf.append(self.mesh.ny) 
            self.zi.append(self.mesh.nz) 
            self.zf.append(self.mesh.nz) 

        ## Define flow region
        # Find which nodes will actually contain a flow and which ones will be in or at a wall
        flowRegion = np.ones((self.mesh.nz, self.mesh.ny, self.mesh.nx)) #essa função tem um comportamento diferente da função TRUE do matlab
        # Add flat plate
        wallJ = np.argmin(np.abs(self.mesh.Y))
        flowRegion[:, 0:wallJ+1, :] = False

        # Add cavities to the flow region
        if hasattr(self.flow_type, 'cav'):
            for i in range(len(self.flow_type.cav)):
                x = self.flow_type.cav[i].x
                y = self.flow_type.cav[i].y
                z = self.flow_type.cav[i].z

                mask_x = (self.mesh.X > x[0]) & (self.mesh.X < x[1])
                mask_y = (self.mesh.Y > y[0]) & (self.mesh.Y < y[1])
                mask_z = (self.mesh.Z > z[0]) & (self.mesh.Z < z[1])

                # Usar numpy.ix_ para aplicar as máscaras booleanas em múltiplas dimensões
                flowRegion[np.ix_(mask_z, mask_y, mask_x)] = True

        # Remove roughnesses from the flow
        if hasattr(self.flow_type, 'rug'):
            for i in range(len(self.flow_type.rug)):
                x = self.flow_type.rug[i].x
                y = self.flow_type.rug[i].y
                z = self.flow_type.rug[i].z

                mask_x = (self.mesh.X >= x[0]) & (self.mesh.X <= x[1])
                mask_y = (self.mesh.Y >= y[0]) & (self.mesh.Y <= y[1])
                mask_z = (self.mesh.Z >= z[0]) & (self.mesh.Z <= z[1])

                flowRegion[np.ix_(mask_z, mask_y, mask_x)] = False
        self.flow_region = flowRegion
        [wallFrontLimits, wallBackLimits, wallUpLimits, wallDownLimits, wallRightLimits, wallLeftLimits, insideWalls, corners] = self.__findWallsForBoundaries(flowRegion)

        ## Create wall for free-slip region
        if self.mesh.X[0] < 0:
            wallUpLimits[:, 0] = np.maximum(wallUpLimits[:, 0], np.argmax(self.mesh.X >= 0))
            wallUpLimits = np.vstack([wallUpLimits, [0, np.where(self.mesh.X < 0)[0][-1], 
                                             np.argmax(self.mesh.Y >= 0), 
                                             np.argmax(self.mesh.Y >= 0), 
                                             0, self.mesh.nz - 1]])

            self.mesh.x.breakPoint = np.array([np.where(self.mesh.X < 0)[0][-1], 
                              np.argmax(self.mesh.Y >= 0), 
                              np.argmax(self.mesh.Y >= 0), 
                              0, self.mesh.nz - 1])

        for i in range(1,7):
            match i:
                case 1:
                    wallPosition = wallFrontLimits
                    wallDir = 'xi'
                case 2:
                    wallPosition = wallBackLimits
                    wallDir = 'xf'  
                case 3:
                    wallPosition = wallUpLimits
                    wallDir = 'yi'
                case 4:
                    wallPosition = wallDownLimits
                    wallDir = 'yf'
                case 5:
                    wallPosition = wallRightLimits
                    wallDir = 'zi'
                case 6:
                    wallPosition = wallLeftLimits
                    wallDir = 'zf'
        
            for j in range(0, wallPosition.shape[0]):
                self.var.append('p')  #%#ok<*SAGROW>
                self.type.append('neu')
                self.dir.append(wallDir)
                self.val.append(0)
                self.xi.append(wallPosition[j,0]+1)
                self.xf.append(wallPosition[j,1]+1)
                self.yi.append(wallPosition[j,2]+1)
                self.yf.append(wallPosition[j,3]+1)
                self.zi.append(wallPosition[j,4]+1)
                self.zf.append(wallPosition[j,5]+1)
        
        ## Add free slip wall, it is the last up facing wall
        if self.mesh.X[0] < 0:
            self.var.append('u')
            self.type.append('neu')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(wallUpLimits[-1,0]+1)
            self.xf.append(wallUpLimits[-1,1]+1)
            self.yi.append(wallUpLimits[-1,2]+1)
            self.yf.append(wallUpLimits[-1,3]+1)
            self.zi.append(wallUpLimits[-1,4]+1)
            self.zf.append(wallUpLimits[-1,5]+1)

            self.var.append('w')
            self.type.append('neu')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(wallUpLimits[-1,0]+1)
            self.xf.append(wallUpLimits[-1,1]+1)
            self.yi.append(wallUpLimits[-1,2]+1)
            self.yf.append(wallUpLimits[-1,3]+1)
            self.zi.append(wallUpLimits[-1,4]+1)
            self.zf.append(wallUpLimits[-1,5]+1)
    
        ## Add regions that are inside walls
        for i in range(0, insideWalls.shape[0]):
            self.var.append('p')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(self.P0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)

            self.var.append('u')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            self.var.append('v')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            self.var.append('w')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            self.var.append('e')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(self.E0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
        
        self.corners = corners
        self.wall = Wall(wallUpLimits, wallDownLimits, wallFrontLimits, wallBackLimits, wallRightLimits, wallLeftLimits)
        self.inside_wall = np.logical_not(flowRegion)

    def __lidDrivenFlow(self):
        ## Define flow region
        # Find which nodes will actually contain a flow and which ones will be in or at a wall
        flowRegion = np.ones((self.mesh.nz, self.mesh.ny, self.mesh.nx))

        # Add cavities to the flow region
        if hasattr(self.flow_type,'cav'):
            for i in range(1,len(self.flow_type.cav)):
                x = self.flow_type.cav[i].x
                y = self.flow_type.cav[i].y
                z = self.flow_type.cav[i].z

                mask_x = (self.mesh.X > x[0]) & (self.mesh.X < x[1])
                mask_y = (self.mesh.Y > y[0]) & (self.mesh.Y < y[1])
                mask_z = (self.mesh.Z > z[0]) & (self.mesh.Z < z[1])

                # Usar numpy.ix_ para aplicar as máscaras booleanas em múltiplas dimensões
                flowRegion[np.ix_(mask_z, mask_y, mask_x)] = True
                
        # Remove roughnesses from the flow
        if hasattr(self.flow_type,'rug'):
            for i in range(1,len(self.flow_type.rug)):
                x = self.flow_type.rug[i].x
                y = self.flow_type.rug[i].y
                z = self.flow_type.rug[i].z

                mask_x = (self.mesh.X >= x[0]) & (self.mesh.X <= x[1])
                mask_y = (self.mesh.Y >= y[0]) & (self.mesh.Y <= y[1])
                mask_z = (self.mesh.Z >= z[0]) & (self.mesh.Z <= z[1])

                flowRegion[np.ix_(mask_z, mask_y, mask_x)] = False
                
        # Add outer walls
        flowRegion[:,:,[0 -1]] = False
        flowRegion[:,[0 -1],:] = False

        self.flow_region = flowRegion
        ## Get walls
        [wallFrontLimits, wallBackLimits, wallUpLimits, wallDownLimits, wallRightLimits, wallLeftLimits, insideWalls, corners] = self.__findWallsForBoundaries(flowRegion)

        ## Add walls to boundary conditions
        for i in range(1,7):
            match i:
                case 1:
                    wallPosition = wallFrontLimits
                    wallDir = 'xi'
                case 2:
                    wallPosition = wallBackLimits
                    wallDir = 'xf'
                case 3:
                    wallPosition = wallUpLimits
                    wallDir = 'yi'
                case 4:
                    wallPosition = wallDownLimits
                    wallDir = 'yf'
                case 5:
                    wallPosition = wallRightLimits
                    wallDir = 'zi'
                case 6:
                    wallPosition = wallLeftLimits
                    wallDir = 'zf'

            for j in range(0, wallPosition.shape[0]):
                self.var.append('p')##ok<*SAGROW>
                self.type.append('neu')
                self.dir.append(wallDir)
                self.val.append(0)
                self.xi.append(wallPosition[j,0]+1)
                self.xf.append(wallPosition[j,1]+1)
                self.yi.append(wallPosition[j,2]+1)
                self.yf.append(wallPosition[j,3]+1)
                self.zi.append(wallPosition[j,4]+1)
                self.zf.append(wallPosition[j,5]+1)
                
        ## Add regions that are inside walls
        for i in range(0, insideWalls.shape[0]):
            self.var.append('p')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(self.P0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)

            self.var.append('u')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            self.var.append('v')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            self.var.append('w')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            if hasattr(self.flow_type,'tWallRelative'):
                eWall = self.E0 * self.flow_ype.tWallRelative
            else:
                eWall = self.E0
            
            self.var.append('e')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(eWall)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)

        self.corners = corners
        self.wall = Wall(wallUpLimits, wallDownLimits, wallFrontLimits, wallBackLimits, wallRightLimits, wallLeftLimits)
        self.inside_wall = np.logical_not(flowRegion)
    
    def __periodicBox(self):
        flowRegion = np.ones((self.mesh.nz, self.mesh.ny, self.mesh.nx))
        [wallFrontLimits, wallBackLimits, wallUpLimits, wallDownLimits, wallRightLimits, wallLeftLimits, insideWalls, corners] = self.__findWallsForBoundaries(flowRegion)

        self.corners = corners
        self.wall = Wall(wallUpLimits, wallDownLimits, wallFrontLimits, wallBackLimits, wallRightLimits, wallLeftLimits)
        self.inside_wall = np.logical_not(flowRegion)

    def __poiseuilleFlow(self):
        ## Define flow region
        # Find which nodes will actually contain a flow and which ones will be in or at a wall
        flowRegion = np.ones((self.mesh.nz, self.mesh.ny, self.mesh.nx))

        # Add cavities to the flow region
        if hasattr(self.flow_type,'cav'):
            for i in range(1,len(self.flow_type.cav)):
                x = self.flow_type.cav[i].x
                y = self.flow_type.cav[i].y
                z = self.flow_type.cav[i].z

                mask_x = (self.mesh.X > x[0]) & (self.mesh.X < x[1])
                mask_y = (self.mesh.Y > y[0]) & (self.mesh.Y < y[1])
                mask_z = (self.mesh.Z > z[0]) & (self.mesh.Z < z[1])

                # Usar numpy.ix_ para aplicar as máscaras booleanas em múltiplas dimensões
                flowRegion[np.ix_(mask_z, mask_y, mask_x)] = True
        # Remove roughnesses from the flow
        if hasattr(self.flow_type,'rug'):
            for i in range(1,len(self.flow_type.rug)):
                x = self.flow_type.rug[i].x
                y = self.flow_type.rug[i].y
                z = self.flow_type.rug[i].z

                mask_x = (self.mesh.X >= x[0]) & (self.mesh.X <= x[1])
                mask_y = (self.mesh.Y >= y[0]) & (self.mesh.Y <= y[1])
                mask_z = (self.mesh.Z >= z[0]) & (self.mesh.Z <= z[1])

                flowRegion[np.ix_(mask_z, mask_y, mask_x)] = False
        # Add outer walls
        flowRegion[:,[0 -1],:] = False

        ## Get walls
        [wallFrontLimits, wallBackLimits, wallUpLimits, wallDownLimits, wallRightLimits, wallLeftLimits, insideWalls, corners] = self.__findWallsForBoundaries(flowRegion)

        ## Add walls to boundary conditions
        for i in range(1,6):
            match i:
                case 1:
                    wallPosition = wallFrontLimits
                    wallDir = 'xi'
                case 2:
                    wallPosition = wallBackLimits
                    wallDir = 'xf'
                case 3:
                    wallPosition = wallUpLimits
                    wallDir = 'yi'
                case 4:
                    wallPosition = wallDownLimits
                    wallDir = 'yf'
                case 5:
                    wallPosition = wallRightLimits
                    wallDir = 'zi'
                case 6:
                    wallPosition = wallLeftLimits
                    wallDir = 'zf'
            
            for j in range(1,wallPosition.shape[0]):
                self.var.append('p')  ##ok<*SAGROW>
                self.type.append('neu')
                self.dir.append(wallDir)
                self.val.append(0)
                self.xi.append(wallPosition[j,0]+1)
                self.xf.append(wallPosition[j,1]+1)
                self.yi.append(wallPosition[j,2]+1)
                self.yf.append(wallPosition[j,3]+1)
                self.zi.append(wallPosition[j,4]+1)
                self.zf.append(wallPosition[j,5]+1)
                
        ## Add regions that are inside walls
        for i in range(1,insideWalls.shape[0]):

            self.var.append('p')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(self.P0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)

            self.var.append('u')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            self.var.append('v')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            self.var.append('w')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(0)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)
            
            if hasattr(self.flow_type,'tWallRelative'):
                eWall = self.E0 * self.flow_type.tWallRelative
            else:
                eWall = self.E0
            
            self.var.append('e')
            self.type.append('dir')
            self.dir.append('yi')
            self.val.append(eWall)
            self.xi.append(insideWalls[i,0]+1)
            self.xf.append(insideWalls[i,1]+1)
            self.yi.append(insideWalls[i,2]+1)
            self.yf.append(insideWalls[i,3]+1)
            self.zi.append(insideWalls[i,4]+1)
            self.zf.append(insideWalls[i,5]+1)

        # Add moving walls
        self.var.append('u')
        self.type.append('dir')
        self.dir.append('yi')
        self.val.append(self.flow_parameters.lowerWallVelocity)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(1)
        self.yf.append(1)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)

        self.var.append('u')
        self.type.append('dir')
        self.dir.append('yf')
        self.val.append(self.flow_parameters.upperWallVelocity)
        self.xi.append(1)
        self.xf.append(self.mesh.nx)
        self.yi.append(self.mesh.ny)
        self.yf.append(self.mesh.ny)
        self.zi.append(1)
        self.zf.append(self.mesh.nz)


        self.corners = corners
        self.wall = Wall(wallUpLimits, wallDownLimits, wallFrontLimits, wallBackLimits, wallRightLimits, wallLeftLimits)
        self.inside_wall = np.logical_not(flowRegion)


    def __findWallsForBoundaries(self, flowRegion):
        # in x (correspondente ao i em Matlab)
        print(flowRegion.shape)
        i = np.arange(1, self.mesh.nx - 1)
        print(max(i))
        #flowRegion[i, :, :] = flowRegion[i, :, :] | (flowRegion[i - 1, :, :] & flowRegion[i + 1, :, :])
        flowRegion[:, :, i] = np.logical_or(flowRegion[:, :, i], np.logical_and(flowRegion[:, :, i - 1], flowRegion[:, :, i + 1]))

        # in y (correspondente ao j em Matlab)
        j = np.arange(1, self.mesh.ny - 1)
        flowRegion[:, j, :] = np.logical_or(flowRegion[:, j, :] , np.logical_and(flowRegion[:, j - 1, :], flowRegion[:, j + 1, :]))

        # in z (correspondente ao k em Matlab)
        k = np.arange(1, self.mesh.nz - 1)
        #flowRegion[:, :, k] = flowRegion[:, :, k] | (flowRegion[:, :, k - 1] & flowRegion[:, :, k + 1])
        flowRegion[k, :, :] = np.logical_or(flowRegion[k, :, :] , np.logical_and(flowRegion[k - 1, :, :] , flowRegion[k + 1, :, :]))

        # Criando as paredes
        #wallFront = np.zeros((self.mesh.nx, self.mesh.ny, self.mesh.nz), dtype=bool)
        wallFront = np.zeros((self.mesh.nz, self.mesh.ny, self.mesh.nx), dtype=bool)
        wallFront[: , :, 0:-1] = np.diff(flowRegion.astype(int), axis=2) == 1

        #wallBack = np.zeros((self.mesh.nx, self.mesh.ny, self.mesh.nz), dtype=bool)
        wallBack = np.zeros((self.mesh.nz, self.mesh.ny, self.mesh.nx), dtype=bool)
        wallBack[:, :, 1:] = np.diff(flowRegion.astype(int), axis=2) == -1

        #wallUp = np.zeros((self.mesh.nx, self.mesh.ny, self.mesh.nz), dtype=bool)
        wallUp = np.zeros((self.mesh.nz, self.mesh.ny, self.mesh.nx), dtype=bool)
        wallUp[:, :-1, :] = np.diff(flowRegion.astype(int), axis=1) == 1

        #wallDown = np.zeros((self.mesh.nx, self.mesh.ny, self.mesh.nz), dtype=bool)
        wallDown = np.zeros((self.mesh.nz, self.mesh.ny, self.mesh.nx), dtype=bool)
        wallDown[:, 1:, :] = np.diff(flowRegion.astype(int), axis=1) == -1

        #wallRight = np.zeros((self.mesh.nx, self.mesh.ny, self.mesh.nz), dtype=bool)
        wallRight = np.zeros((self.mesh.nz, self.mesh.ny, self.mesh.nx), dtype=bool)
        wallRight[:-1, :, :] = np.diff(flowRegion.astype(int), axis=0) == 1

        #wallLeft = np.zeros((self.mesh.nx, self.mesh.ny, self.mesh.nz), dtype=bool)
        wallLeft = np.zeros((self.mesh.nz, self.mesh.ny, self.mesh.nx), dtype=bool)
        wallLeft[1:, :, :] = np.diff(flowRegion.astype(int), axis=0) == -1

        # Encontra os limites das paredes para as fronteiras
        #wallFrontLimits, wallBackLimits = self._find_wall_limits(self.mesh.nx, wallFront, wallBack, axis=2)
        #wallUpLimits, wallDownLimits = self._find_wall_limits(self.mesh.ny, wallUp, wallDown, axis=1)
        #wallRightLimits, wallLeftLimits = self._find_wall_limits(self.mesh.nz, wallRight, wallLeft)
        wallFrontLimits, wallBackLimits, wallUpLimits, wallDownLimits, wallRightLimits, wallLeftLimits = self._find_wall_limits(wallFront, wallBack, wallUp, wallDown, wallRight, wallLeft)


        # Mesclando paredes adjacentes
        wallFrontLimits = self._merge_adjacent_walls(wallFrontLimits)
        wallBackLimits = self._merge_adjacent_walls(wallBackLimits)
        wallUpLimits = self._merge_adjacent_walls(wallUpLimits)
        wallDownLimits = self._merge_adjacent_walls(wallDownLimits)

        # Encontrando os cantos (com duas ou três paredes)
        corners = self._find_corners(wallFront, wallBack, wallUp, wallDown, wallRight, wallLeft)

        # Encontrando regiões completamente contidas dentro das paredes
        insideWalls = self._find_inside_walls(flowRegion)

        return np.array(wallFrontLimits), np.array(wallBackLimits), np.array(wallUpLimits), np.array(wallDownLimits), np.array(wallRightLimits), np.array(wallLeftLimits), insideWalls, corners
    
    def _find_wall_limits(self, wallFront, wallBack, wallUp, wallDown, wallRight, wallLeft):
        wallFrontLimits = []
        wallBackLimits = []
        wallUpLimits = []
        wallDownLimits = []
        wallRightLimits = []
        wallLeftLimits = []

        #Front and Back
        for i in range(self.mesh.nx):
            localWall = np.take(wallFront, i, axis=2)  # Acessa como 2D
            if np.any(localWall):
                wallStarts = []
                wallEnds = []
                kStart = 0
                for k in range(self.mesh.nz):
                    diffWall = np.diff(localWall[k, :].astype(int), axis=0)
                    if k == 0 or (len(wallStarts) == 0):
                        wallStarts = np.where(np.concatenate(([localWall[k, 0]], diffWall == 1)))[0]
                        wallEnds = np.where(np.concatenate((diffWall == -1, [localWall[k, -1]])))[0]
                        kStart = k

                    if k == self.mesh.nz - 1 or np.any(localWall[k, :] != localWall[k + 1, :]):
                        for j in range(len(wallStarts)):
                            wallFrontLimits.append([i, i, wallStarts[j], wallEnds[j], kStart, k])
                        wallStarts = []
                        wallEnds = []
            localWall = np.take(wallBack, i, axis=2)
            if np.any(localWall):
                wallStarts = []
                wallEnds = []
                kStart = 0
                for k in range(self.mesh.nz):
                    diffWall = np.diff(localWall[k, :].astype(int), axis=0)
                    if k == 0 or (len(wallStarts) == 0):
                        wallStarts = np.where(np.concatenate(([localWall[k, 0]], diffWall == 1)))[0]
                        wallEnds = np.where(np.concatenate((diffWall == -1, [localWall[k, -1]])))[0]
                        print('wallEnds:')
                        print(wallEnds)
                        kStart = k
                        
                    if k == self.mesh.nz - 1 or np.any(localWall[k, :] != localWall[k + 1, :]):
                        for j in range(len(wallStarts)):
                            wallBackLimits.append([i, i, wallStarts[j], wallEnds[j], kStart, k])
                        wallStarts = []
                        wallEnds = []

        #Up and Down
        for j in range(self.mesh.ny):
            localWall = np.take(wallUp, j, axis = 1)
            if np.any(localWall):
                wallStarts = []
                wallEnds = []
                kStart = 0
                for k in range(self.mesh.nz):
                    diffWall = np.diff(localWall[k, :].astype(int), axis = 0)
                    if k == 0 or (len(wallStarts) == 0):
                        wallStarts = np.where(np.concatenate(([localWall[k, 0]], diffWall == 1)))[0]
                        wallEnds = np.where(np.concatenate((diffWall == -1, [localWall[k, -1]])))[0]
                        kStart = k
                        
                    if k == self.mesh.nz - 1 or np.any(localWall[k, :] != localWall[k + 1, :]):
                        for i in range(len(wallStarts)):
                            wallUpLimits.append([wallStarts[i], wallEnds[i], j, j, kStart, k])
                        wallStarts = []
                        wallEnds = []

            localWall = np.take(wallDown, j, axis = 1)
            if np.any(localWall):
                wallStarts = []
                wallEnds = []
                kStart = 0
                for k in range(self.mesh.nz):
                    diffWall = np.diff(localWall[k, :].astype(int), axis = 0)
                    if k == 0 or (len(wallStarts) == 0):
                        wallStarts = np.where(np.concatenate(([localWall[k, 0]], diffWall == 1)))[0]
                        wallEnds = np.where(np.concatenate((diffWall == -1, [localWall[k, -1]])))[0]
                        kStart = k
                        
                    if k == self.mesh.nz - 1 or np.any(localWall[k, :] != localWall[k + 1, :]):
                        for i in range(len(wallStarts)):
                            wallDownLimits.append([wallStarts[i], wallEnds[i], j, j, kStart, k])
                        wallStarts = []
                        wallEnds = []

        #Right and Left
        for k in range(self.mesh.nz):
            localWall = np.take(wallRight, k, axis = 0)
            if np.any(localWall):
                wallStarts = []
                wallEnds = []
                iStart = 0
                for i in range(self.mesh.nx):
                    diffWall = np.diff(localWall[i, :].astype(int), axis = 0)
                    if i == 0 or (len(wallStarts) == 0):
                        wallStarts = np.where(np.concatenate(([localWall[i, 0]], diffWall == 1)))[0]
                        wallEnds = np.where(np.concatenate((diffWall == -1, [localWall[i, -1]])))[0]
                        iStart = i
                        
                    if i == self.mesh.nx - 1 or np.any(localWall[i, :] != localWall[i + 1, :]):
                        for j in range(len(wallStarts)):
                            wallRightLimits.append([iStart, i, wallStarts[j], wallEnds[j], k, k])
                        wallStarts = []
                        wallEnds = []

            localWall = np.take(wallLeft, k, axis = 0)
            if np.any(localWall):
                wallStarts = []
                wallEnds = []
                iStart = 0
                for i in range(self.mesh.nx):
                    diffWall = np.diff(localWall[i, :].astype(int), axis = 0)
                    if i == 0 or (len(wallStarts) == 0):
                        wallStarts = np.where(np.concatenate(([localWall[i, 0]], diffWall == 1)))[0]
                        wallEnds = np.where(np.concatenate((diffWall == -1, [localWall[i, -1]])))[0]
                        iStart = i
                        
                    if i == self.mesh.nx - 1 or np.any(localWall[i, :] != localWall[i + 1, :]):
                        for j in range(len(wallStarts)):
                            wallLeftLimits.append([iStart, i, wallStarts[j], wallEnds[j], k, k])
                        wallStarts = []
                        wallEnds = []

        
        return wallFrontLimits, wallBackLimits, wallUpLimits, wallDownLimits, wallRightLimits, wallLeftLimits


    def _find_wall_limits22(self, wallFront, wallBack, wallUp, wallDown, wallRight, wallLeft):
        # Limites de cada tipo de parede
        wallFrontLimits = []
        wallBackLimits = []
        wallUpLimits = []
        wallDownLimits = []
        wallRightLimits = []
        wallLeftLimits = []

        # Função auxiliar para processar as paredes (frontal/traseira, superior/inferior, direita/esquerda)
        def process_wall(wallType, axis, limits, nx_or_ny):
            for i in range(nx_or_ny):
                localWall = np.take(wallType, i, axis=axis)  # Acessa como 2D
                if np.any(localWall):
                    wallStarts = []
                    wallEnds = []
                    kStart = 0
                    for k in range(self.mesh.nz):
                        diffWall = np.diff(localWall[k, :], axis=0)

                        # Identificar os pontos de início e fim da parede (mudança de 0 para 1 e 1 para 0)
                        wallStarts = np.where(np.concatenate(([localWall[k, 0]], diffWall == 1)))[0]
                        wallEnds = np.where(np.concatenate((diffWall == -1, [localWall[k, -1]])))[0]

                        if k == 0 or not wallStarts.size:
                            kStart = k

                        if k == self.mesh.nz - 1 or np.any(localWall[k, :] != localWall[k + 1, :]):
                            for j in range(len(wallStarts)):
                                limits.append([i, i, wallStarts[j], wallEnds[j], kStart, k])
                            wallStarts = []
                            wallEnds = []

        # Processando as paredes frontais e traseiras
        process_wall(wallFront, axis=2, limits=wallFrontLimits, nx_or_ny=self.mesh.nx)
        process_wall(wallBack, axis=2, limits=wallBackLimits, nx_or_ny=self.mesh.nx)

        # Processando as paredes superiores e inferiores
        process_wall(wallUp, axis=1, limits=wallUpLimits, nx_or_ny=self.mesh.ny)
        process_wall(wallDown, axis=1, limits=wallDownLimits, nx_or_ny=self.mesh.ny)

        # Processando as paredes direita e esquerda
        process_wall(wallRight, axis=0, limits=wallRightLimits, nx_or_ny=self.mesh.nz)
        process_wall(wallLeft, axis=0, limits=wallLeftLimits, nx_or_ny=self.mesh.nz)

        # Retorna todas as variáveis de limite
        return wallFrontLimits, wallBackLimits, wallUpLimits, wallDownLimits, wallRightLimits, wallLeftLimits
    
    
    
    
    
    def _find_wall_limits2(self, size, wall1, wall2, axis=0):
        wallLimits1 = []
        wallLimits2 = []

        for i in range(size):
            localWall = wall1.take(i, axis=axis)  # Correspondente ao wallFront(i,:,:) no Matlab
            if np.any(localWall):
                wallLimits1 += self._get_wall_limits(localWall, i, axis)

            localWall = wall2.take(i, axis=axis)  # Correspondente ao wallBack(i,:,:) no Matlab
            if np.any(localWall):
                wallLimits2 += self._get_wall_limits(localWall, i, axis)

        return wallLimits1, wallLimits2

    def _get_wall_limits2(self, localWall, i, axis):
        wallLimits = []
        wallStarts = np.where([localWall.take(0, axis=axis)] + np.diff(localWall, axis=axis) == 1)[0]
        wallEnds = np.where([np.diff(localWall, axis=axis) == -1] + [localWall.take(-1, axis=axis)])[0]

        kStart = 0
        for k in range(localWall.shape[axis]):
            if k == 0 or not wallStarts:
                wallStarts = np.where([localWall.take(0, axis=axis)] + np.diff(localWall, axis=axis) == 1)[0]
                wallEnds = np.where([np.diff(localWall, axis=axis) == -1] + [localWall.take(-1, axis=axis)])[0]
                kStart = k
            if k == localWall.shape[axis] - 1 or np.any(localWall != localWall.take(k + 1, axis=axis)):
                for start, end in zip(wallStarts, wallEnds):
                    wallLimits.append([i, i, start, end, kStart, k])
                wallStarts = []
                wallEnds = []
        return wallLimits

    def _merge_adjacent_walls(self, wallLimits):
        done = False
        while not done:
            done = True
            nWalls = len(wallLimits)
            for i in range(nWalls - 1):
                for j in range(i + 1, nWalls):
                    if np.all(wallLimits[i][:4] == wallLimits[j][:4]) and (wallLimits[i][5] == (wallLimits[j][4] - 1)):
                        toMerge = [i, j]
                        done = False
                        break
                if not done:
                    break
            if not done:
                wallLimits[toMerge[0]][5] = wallLimits[toMerge[1]][5]
                wallLimits.pop(toMerge[1])
        return wallLimits

    def _find_corners(self, wallFront, wallBack, wallUp, wallDown, wallRight, wallLeft):
        
        corners = Corners()
        
        # With 2 walls
        corners_matrix = np.zeros((self.mesh.nz, self.mesh.ny, self.mesh.nx, 4), dtype=bool)
        
        # Constant z
        corners_matrix[:, :, :, 0] = wallFront & wallUp
        corners_matrix[:, :, :, 1] = wallFront & wallDown
        corners_matrix[:, :, :, 2] = wallBack & wallUp
        corners_matrix[:, :, :, 3] = wallBack & wallDown
        corner_directions = np.array([[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]])
        
        for i in range(self.mesh.nx):
            for j in range(self.mesh.ny):
                for m in range(4):
                    corner_row = corners_matrix[:, j, i, m]
                    if np.any(corner_row):
                        corner_starts = np.where(np.concatenate(([corner_row[0]], np.diff(corner_row.astype(int)) == 1)))[0]
                        corner_ends = np.where(np.concatenate((np.diff(corner_row.astype(int)) == -1, [corner_row[-1]])))[0]
                        for n in range(len(corner_starts)):
                            corners.limits.append([i+1, i+1, j+1, j+1, corner_starts[n]+1, corner_ends[n]+1])
                            corners.dir.append(corner_directions[m, :])

        # Constant x
        corners_matrix[:, :, :, 0] = wallFront & wallRight
        corners_matrix[:, :, :, 1] = wallFront & wallLeft
        corners_matrix[:, :, :, 2] = wallBack & wallRight
        corners_matrix[:, :, :, 3] = wallBack & wallLeft
        corner_directions = np.array([[1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]])
        
        for i in range(self.mesh.nx):
            for k in range(self.mesh.nz):
                for m in range(4):
                    corner_row = corners_matrix[k, :, i, m]
                    if np.any(corner_row):
                        corner_starts = np.where(np.concatenate(([corner_row[0]], np.diff(corner_row.astype(int)) == 1)))[0]
                        corner_ends = np.where(np.concatenate((np.diff(corner_row.astype(int)) == -1, [corner_row[-1]])))[0]
                        for n in range(len(corner_starts)):
                            corners.limits.append([i+1, i+1, corner_starts[n]+1, corner_ends[n]+1, k+1, k+1])
                            corners.dir.append(corner_directions[m, :])

        # Constant y
        corners_matrix[:, :, :, 0] = wallUp & wallRight
        corners_matrix[:, :, :, 1] = wallUp & wallLeft
        corners_matrix[:, :, :, 2] = wallDown & wallRight
        corners_matrix[:, :, :, 3] = wallDown & wallLeft
        corner_directions = np.array([[0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]])
        
        for j in range(self.mesh.ny):
            for k in range(self.mesh.nz):
                for m in range(4):
                    corner_row = corners_matrix[k, j, :, m]
                    if np.any(corner_row):
                        corner_starts = np.where(np.concatenate(([corner_row[0]], np.diff(corner_row.astype(int)) == 1)))[0]
                        corner_ends = np.where(np.concatenate((np.diff(corner_row.astype(int)) == -1, [corner_row[-1]])))[0]
                        for n in range(len(corner_starts)):
                            corners.limits.append([corner_starts[n]+1, corner_ends[n]+1, j+1, j+1, k+1, k+1])
                            corners.dir.append(corner_directions[m, :])

        # With 3 walls
        corners_matrix = np.zeros((self.mesh.nz, self.mesh.ny, self.mesh.nx, 8), dtype=bool)

        corners_matrix[:, :, :, 0] = wallFront & wallUp & wallRight
        corners_matrix[:, :, :, 1] = wallFront & wallUp & wallLeft
        corners_matrix[:, :, :, 2] = wallFront & wallDown & wallRight
        corners_matrix[:, :, :, 3] = wallFront & wallDown & wallLeft
        corners_matrix[:, :, :, 4] = wallBack & wallUp & wallRight
        corners_matrix[:, :, :, 5] = wallBack & wallUp & wallLeft
        corners_matrix[:, :, :, 6] = wallBack & wallDown & wallRight
        corners_matrix[:, :, :, 7] = wallBack & wallDown & wallLeft

        corner_directions = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                                      [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]])

        I, J, K, M = np.where(corners_matrix)
        for i, j, k, m in zip(I, J, K, M):
            corners.limits.append([i+1, i+1, j+1, j+1, k+1, k+1])
            corners.dir.append(corner_directions[m, :])

        corners.adiabatic = np.zeros((len(corners.dir), 1))

        return corners

    def _find_inside_walls(self, flowRegion):
        isWall = np.logical_not(flowRegion)

        insideWalls = []
        for k in range(self.mesh.nz):
            for j in range(self.mesh.ny):
                wallBegins = np.where(np.concatenate(([isWall[k, j, 0]], np.diff(isWall[k, j, :].astype(int)) == 1)))[0]
                wallEnds = np.where(np.concatenate((np.diff(isWall[k, j, :].astype(int)) == -1, [isWall[k, j, -1]])))[0]
                for i in range(len(wallBegins)):
                    insideWalls.append([wallBegins[i], wallEnds[i], j, j, k, k])

        insideWalls = self._sort_and_merge_walls(insideWalls)
        return insideWalls
    
    def _get_unique_indices(arr):
        unique_rows = {}
        inverse = []
        for i, row in enumerate(arr):
            key = tuple(row)
            if key not in unique_rows:
                unique_rows[key] = len(unique_rows)
            inverse.append(unique_rows[key])
        return inverse

    def _sort_and_merge_walls4(self, insideWalls):
        if len(insideWalls) > 0:
        
            # Get indices based on first two columns
            indices = self._get_unique_indices(insideWalls[:, [0, 1]])
        
            # Sort by indices
            sort_indices = sorted(range(len(indices)), key=lambda k: indices[k])
            insideWalls = insideWalls[sort_indices]
        
            # Merge limits in y
            i = 0
            while i < len(insideWalls) - 1:
                if (all(insideWalls[i, [0,1,4,5]] == insideWalls[i+1, [0,1,4,5]]) and 
                    insideWalls[i, 3] + 1 == insideWalls[i+1, 2]):
                    insideWalls[i, 3] = insideWalls[i+1, 3]
                    insideWalls = np.delete(insideWalls, i+1, axis=0)
                else:
                    i += 1

            # Merge limits in z
            i = 0
            while i < len(insideWalls) - 1:
                if (all(insideWalls[i, [0,1,2,3]] == insideWalls[i+1, [0,1,2,3]]) and 
                    insideWalls[i, 5] + 1 == insideWalls[i+1, 4]):
                    insideWalls[i, 5] = insideWalls[i+1, 5]
                    insideWalls = np.delete(insideWalls, i+1, axis=0)
                else:
                    i += 1

        return insideWalls
    def _sort_and_merge_walls3(self, insideWalls):
        if len(insideWalls) > 0:
            # Sort by wall limits
            _, unique_indices = np.unique(insideWalls[:, [0, 1]], axis=0, return_inverse=True)
            sort_indices = np.argsort(unique_indices)
            insideWalls = insideWalls[sort_indices]
        
            # Merge limits in y
            i = 0
            while i < len(insideWalls) - 1:
                if (np.array_equal(insideWalls[i, [0, 1, 4, 5]], insideWalls[i+1, [0, 1, 4, 5]]) and 
                    insideWalls[i, 3] + 1 == insideWalls[i+1, 2]):
                    insideWalls[i, 3] = insideWalls[i+1, 3]
                    insideWalls = np.delete(insideWalls, i+1, axis=0)
                else:
                    i += 1

            # Merge limits in z
            i = 0
            while i < len(insideWalls) - 1:
                if (np.array_equal(insideWalls[i, [0, 1, 2, 3]], insideWalls[i+1, [0, 1, 2, 3]]) and 
                    insideWalls[i, 5] + 1 == insideWalls[i+1, 4]):
                    insideWalls[i, 5] = insideWalls[i+1, 5]
                    insideWalls = np.delete(insideWalls, i+1, axis=0)
                else:
                    i += 1

            return insideWalls

    def _sort_and_merge_walls2(self, insideWalls):
        if np.array(insideWalls).size > 0:
            insideWalls = np.array(insideWalls)
            unique_rows, ind = np.unique(insideWalls[:, [0]], axis=0, return_inverse=True)
            ind +=1
            ind[0]-=1
            sorted_indices = np.argsort(ind)
            insideWalls = insideWalls[sorted_indices, :]
        return self._merge_walls_by_dimension(insideWalls)

    def _sort_and_merge_walls2(self, insideWalls):
        if np.array(insideWalls).size > 0:
            dic = {}; idxs = []; idx = 0
            insideWalls = np.array(insideWalls)
            
            for l in insideWalls:
                if tuple(l[[0, 1]]) not in dic:
                    dic[tuple(l[[0, 1]])] = idx
                    idxs.append(idx)
                    idx = idx + 1
                else:
                    idxs.append(dic[tuple(l[[0, 1]])])

        return insideWalls[np.argsort(idxs, stable=True), :]

    def _sort_and_merge_walls(self, insideWalls):
        insideWalls = self._sort_and_merge_walls2(insideWalls)
        insideWalls = self._merge_walls_by_dimension(insideWalls)
        return insideWalls
    
        if np.array(insideWalls).size > 0:
            insideWalls = np.array(insideWalls)
            unique_rows, ind, ind2 = np.unique(insideWalls[:, [0, 1]], axis=0, return_index=True, return_inverse=True)
            insideWalls = insideWalls[np.sort(ind), :]
        return np.array(insideWalls)

    def _merge_walls_by_dimension(self, insideWalls):
        i = 0
        while i < len(insideWalls) - 1:
            if (np.array_equal(insideWalls[i, [0, 1, 4, 5]], insideWalls[i+1, [0, 1, 4, 5]]) and 
                insideWalls[i, 3] + 1 == insideWalls[i+1, 2]):
                insideWalls[i, 3] = insideWalls[i+1, 3]
                insideWalls = np.delete(insideWalls, i+1, axis=0)
            else:
                i += 1

        # Merge limits in z
        i = 0
        while i < len(insideWalls) - 1:
            if (np.array_equal(insideWalls[i, [0, 1, 2, 3]], insideWalls[i+1, [0, 1, 2, 3]]) and 
                insideWalls[i, 5] + 1 == insideWalls[i+1, 4]):
                insideWalls[i, 5] = insideWalls[i+1, 5]
                insideWalls = np.delete(insideWalls, i+1, axis=0)
            else:
                i += 1
        return insideWalls


    def get_neumann_coeffs(self, neumann_order):
        NC1 = [
            [-1, 1], [-3 / 2, 2, -1 / 2], [-11 / 6, 3, -3 / 2, 1 / 3],
            [-25 / 12, 4, -3, 4 / 3, -1 / 4], [-137 / 60, 5, -5, 10 / 3, -5 / 4, 1 / 5],
            [-49 / 20, 6, -15 / 2, 20 / 3, -15 / 4, 6 / 5, -1 / 6]
        ]
        NC2 = [
            [1, -2, 1], [2, -5, 4, -1], [35 / 12, -26 / 3, 19 / 2, -14 / 3, 11 / 12],
            [15 / 4, -77 / 6, 107 / 6, -13, 61 / 12, -5 / 6], [203 / 45, -87 / 5, 117 / 4, -254 / 9, 33 / 2, -27 / 5, 137 / 180],
            [469 / 90, -223 / 10, 879 / 20, -949 / 18, 41, -201 / 10, 1019 / 180, -7 / 10]
        ]

        self.neumann_coeffs = -np.array(NC1[neumann_order[0]-1][1:]) / NC1[neumann_order[0]-1][0]
        self.neumann2_coeffs = -np.array(NC2[neumann_order[1]-1][1:]) / NC2[neumann_order[1]-1][0]

    def add_disturbances(self):
        if hasattr(self.flow_type, 'disturb'):
            for disturb in self.flow_type.disturb:
                if disturb.active:
                    disturb_info = {
                        'type': disturb.type,
                        'forcing': getattr(disturb, 'forcing', False),
                        'par': disturb.par if hasattr(disturb, 'par') else [],
                        'var': disturb.var
                    }

                    disturb_info = deepcopy(disturb)

                    # Handling indices
                    xi, xf = disturb.x[0], disturb.x[1]
                    xi = self.mesh.X[0] if xi == -np.inf else (self.mesh.X[-1] if xi == np.inf else xi)
                    xf = self.mesh.X[0] if xf == -np.inf else (self.mesh.X[-1] if xf == np.inf else xf)

                    yi, yf = disturb.y[0], disturb.y[1]
                    yi = self.mesh.Y[0] if yi == -np.inf else (self.mesh.Y[-1] if yi == np.inf else yi)
                    yf = self.mesh.Y[0] if yf == -np.inf else (self.mesh.Y[-1] if yf == np.inf else yf)

                    zi, zf = disturb.z[0], disturb.z[1]
                    zi = self.mesh.Z[0] if zi == -np.inf else (self.mesh.Z[-1] if zi == np.inf else zi)
                    zf = self.mesh.Z[0] if zf == -np.inf else (self.mesh.Z[-1] if zf == np.inf else zf)

                    # Find indices in the mesh
                    ind = [
                        np.where(self.mesh.X>=xi)[0][0],
                        np.where(self.mesh.X<=xf)[0][-1],
                        np.where(self.mesh.Y>=yi)[0][0],
                        np.where(self.mesh.Y<=yf)[0][-1],
                        np.where(self.mesh.Z>=zi)[0][0],
                        np.where(self.mesh.Z<=zf)[0][-1]
                    ]
                    ind = ind + np.array(1)

                    disturb_info.ind = ind

                    if not hasattr(disturb, 'extraNodes'):
                        disturb_info.extraNodes = [0, 0, 0, 0, 0, 0]

                    for i in range(6):
                        ind[i] -= disturb_info.extraNodes[i] if i % 2 == 0 else +disturb_info.extraNodes[i]

                    self.disturb.append(disturb_info)

    def init_boundaries(self, boundary_data, domain_slices_y, domain_slices_z, p_row, p_col):
        self.nUd = 0
        self.nVd = 0
        self.nWd = 0
        self.nPd = 0
        self.nEd = 0
        self.nUn = 0
        self.nVn = 0
        self.nWn = 0
        self.nPn = 0
        self.nEn = 0
        self.nUs = 0
        self.nVs = 0
        self.nWs = 0
        self.nPs = 0
        self.nEs = 0
        
        self.iUd = []
        self.iVd = []
        self.iWd = []
        self.iPd = []
        self.iEd = []
        self.iUn = []
        self.iVn = []
        self.iWn = []
        self.iPn = []
        self.iEn = []
        self.iUs = []
        self.iVs = []
        self.iWs = []
        self.iPs = []
        self.iEs = []
        
        self.vUd = []
        self.vVd = []
        self.vWd = []
        self.vPd = []
        self.vEd = []
        self.dUn = []
        self.dVn = []
        self.dWn = []
        self.dPn = []
        self.dEn = []
        self.dUs = []
        self.dVs = []
        self.dWs = []
        self.dPs = []
        self.dEs = []

        self.cL = boundary_data['corners']['limits']
        self.cD = boundary_data['corners']['dir']
        self.adiabatic = boundary_data['corners']['adiabatic']
        self.cN = self.cL.shape[0]
        
        self.gamma1 = boundary_data['gamma'] - 1
        self.E0 = boundary_data['E0']

        self.bi = [self.initialize_boundary() for _ in range(p_row * p_col)]
        
        self.__init_boundaries(boundary_data, domain_slices_y, domain_slices_z, p_row, p_col)
        
    def initialize_boundary(self):
        return {
            'nUd': 0, 'nVd': 0, 'nWd': 0, 'nPd': 0, 'nEd': 0,
            'nUn': 0, 'nVn': 0, 'nWn': 0, 'nPn': 0, 'nEn': 0,
            'nUs': 0, 'nVs': 0, 'nWs': 0, 'nPs': 0, 'nEs': 0,
            'iUd': [], 'iVd': [], 'iWd': [], 'iPd': [], 'iEd': [],
            'iUn': [], 'iVn': [], 'iWn': [], 'iPn': [], 'iEn': [],
            'iUs': [], 'iVs': [], 'iWs': [], 'iPs': [], 'iEs': [],
            'vUd': [], 'vVd': [], 'vWd': [], 'vPd': [], 'vEd': [],
            'dUn': [], 'dVn': [], 'dWn': [], 'dPn': [], 'dEn': [],
            'dUs': [], 'dVs': [], 'dWs': [], 'dPs': [], 'dEs': [],
            'disturb': []
        }
    def __init_boundaries(self, boundary_data, domain_slices_y, domain_slices_z, p_row, p_col):
        direction_order = ['xi', 'xf', 'yi', 'yf', 'zi', 'zf']
        
        for i in range(len(boundary_data['val'])):
            if boundary_data['type'][i] == 'dir':
                self.process_dir_boundary(boundary_data, i)
            elif boundary_data['type'][i] == 'neu':
                self.process_neu_boundary(boundary_data, i, direction_order)
            elif boundary_data['type'][i] == 'sec':
                self.process_sec_boundary(boundary_data, i, direction_order)

        self.split_boundaries(self, domain_slices_y, domain_slices_z, p_row, p_col)

    def process_dir_boundary(self, boundary_data, i):
        var = boundary_data['var'][i]
        if var == 'u':
            self.nUd += 1
            self.iUd.append(self.get_boundary_indices(boundary_data, i))
            self.vUd.append(boundary_data['val'][i])
        elif var == 'v':
            self.nVd += 1
            self.iVd.append(self.get_boundary_indices(boundary_data, i))
            self.vVd.append(boundary_data['val'][i])
        elif var == 'w':
            self.nWd += 1
            self.iWd.append(self.get_boundary_indices(boundary_data, i))
            self.vWd.append(boundary_data['val'][i])
        elif var == 'p':
            self.nPd += 1
            self.iPd.append(self.get_boundary_indices(boundary_data, i))
            self.vPd.append(boundary_data['val'][i])
        elif var == 'e':
            self.nEd += 1
            self.iEd.append(self.get_boundary_indices(boundary_data, i))
            self.vEd.append(boundary_data['val'][i])

    def process_neu_boundary(self, boundary_data, i, direction_order):
        var = boundary_data['var'][i]
        if var == 'u':
            self.nUn += 1
            self.iUn.append(self.get_boundary_indices(boundary_data, i))
            self.dUn.append(direction_order.index(boundary_data['dir'][i]))
        elif var == 'v':
            self.nVn += 1
            self.iVn.append(self.get_boundary_indices(boundary_data, i))
            self.dVn.append(direction_order.index(boundary_data['dir'][i]))
        elif var == 'w':
            self.nWn += 1
            self.iWn.append(self.get_boundary_indices(boundary_data, i))
            self.dWn.append(direction_order.index(boundary_data['dir'][i]))
        elif var == 'p':
            self.nPn += 1
            self.iPn.append(self.get_boundary_indices(boundary_data, i))
            self.dPn.append(direction_order.index(boundary_data['dir'][i]))
        elif var == 'e':
            self.nEn += 1
            self.iEn.append(self.get_boundary_indices(boundary_data, i))
            self.dEn.append(direction_order.index(boundary_data['dir'][i]))

    def process_sec_boundary(self, boundary_data, i, direction_order):
        var = boundary_data['var'][i]
        if var == 'u':
            self.nUs += 1
            self.iUs.append(self.get_boundary_indices(boundary_data, i))
            self.dUs.append(direction_order.index(boundary_data['dir'][i]))
        elif var == 'v':
            self.nVs += 1
            self.iVs.append(self.get_boundary_indices(boundary_data, i))
            self.dVs.append(direction_order.index(boundary_data['dir'][i]))
        elif var == 'w':
            self.nWs += 1
            self.iWs.append(self.get_boundary_indices(boundary_data, i))
            self.dWs.append(direction_order.index(boundary_data['dir'][i]))
        elif var == 'p':
            self.nPs += 1
            self.iPs.append(self.get_boundary_indices(boundary_data, i))
            self.dPs.append(direction_order.index(boundary_data['dir'][i]))
        elif var == 'e':
            self.nEs += 1
            self.iEs.append(self.get_boundary_indices(boundary_data, i))
            self.dEs.append(direction_order.index(boundary_data['dir'][i]))

    def get_boundary_indices(self, boundary_data, i):
        return [
            boundary_data['xi'][i],
            boundary_data['xf'][i],
            boundary_data['yi'][i],
            boundary_data['yf'][i],
            boundary_data['zi'][i],
            boundary_data['zf'][i]
        ]
    #TODO: Revisar a implementação deste método
    def split_boundaries(self, domain_slices_y, domain_slices_z, p_row, p_col):
        for j in range(p_row):
            for k in range(p_col):
                nProc = k + j * p_col
                biL = self.bi[nProc]

                Ji = domain_slices_y[0][j]
                Jf = domain_slices_y[1][j]
                Ki = domain_slices_z[0][k]
                Kf = domain_slices_z[1][k]

                for var in ['Ud', 'Vd', 'Wd', 'Pd', 'Ed']:
                    biL[f'i{var}'], biL[f'n{var}'], biL[f'v{var}'] = self.limit_indices(
                        biL[f'i{var}'], biL[f'n{var}'], biL[f'v{var}'], 'd', Ji, Jf, Ki, Kf, 0
                    )

                for var in ['Un', 'Vn', 'Wn', 'Pn', 'En']:
                    biL[f'i{var}'], biL[f'n{var}'], biL[f'd{var}'] = self.limit_indices(
                        biL[f'i{var}'], biL[f'n{var}'], biL[f'd{var}'], 'n', Ji, Jf, Ki, Kf, 0
                    )

                for var in ['Us', 'Vs', 'Ws', 'Ps', 'Es']:
                    biL[f'i{var}'], biL[f'n{var}'], biL[f'd{var}'] = self.limit_indices(
                        biL[f'i{var}'], biL[f'n{var}'], biL[f'd{var}'], 'n', Ji, Jf, Ki, Kf, 0
                    )

                values = biL['cD'] + [biL['adiabatic']]
                biL['cL'], biL['cN'], values = self.limit_indices(
                    biL['cL'], biL['cN'], values, 'c', Ji, Jf, Ki, Kf, 0
                )
                biL['cD'] = values[:, :3]
                biL['adiabatic'] = values[:, 3]

        # Split disturbances across processors
        for j in range(p_row):
            for k in range(p_col):
                nProc = k + j * p_col
                disturb = []
                for i in range(len(self.disturb)):
                    if self.disturb[i]:
                        ind = self.disturb[i]['ind']
                        ind[2:6] = [
                            max(ind[2], domain_slices_y[0][j]),
                            min(ind[3], domain_slices_y[1][j]),
                            max(ind[4], domain_slices_z[0][k]),
                            min(ind[5], domain_slices_z[1][k])
                        ]

                        if ind[2] <= ind[3] and ind[4] <= ind[5]:
                            disturb.append(self.disturb[i])
                            disturb[-1]['ind'] = ind
                            disturb[-1]['X'] = self.mesh['X'][ind[0]:ind[1]]
                            disturb[-1]['Y'] = self.mesh['Y'][ind[2]:ind[3]]
                            disturb[-1]['Z'] = self.mesh['Z'][ind[4]:ind[5]]

                self.bi[nProc]['disturb'] = disturb

    def limit_indices(self, ind, n, vd, type_, Ji, Jf, Ki, Kf):
        if n == 0:
            return ind, n, vd
        
        for i in range(n):
            ind[i][2:6] = [max(ind[i][2], Ji), min(ind[i][3], Jf),
                           max(ind[i][4], Ki), min(ind[i][5], Kf)]
        
        to_remove = (ind[:, 2] > ind[:, 3]) | (ind[:, 4] > ind[:, 5])
        ind = ind[~to_remove]
        if len(vd) == 1 and n > 1:
            vd = vd[~to_remove]
        else:
            vd = vd[~to_remove]

        n -= np.sum(to_remove)
        
        if type_ == 'n':
            for i in range(n):
                if vd[i] == 3 and ind[i][3] + 0 > Jf:
                    raise ValueError(f'There is a y+ Neumann condition at J = {ind[i][3]} crossing a domain slice at J = {Jf}. Consider changing p_row.')
                elif vd[i] == 4 and ind[i][2] - 0 < Ji:
                    raise ValueError(f'There is a y- Neumann condition at J = {ind[i][2]} crossing a domain slice at J = {Ji}. Consider changing p_row.')
                elif vd[i] == 5 and ind[i][5] + 0 > Kf:
                    raise ValueError(f'There is a z+ Neumann condition at K = {ind[i][5]} crossing a domain slice at K = {Kf}. Consider changing p_col.')
                elif vd[i] == 6 and ind[i][4] - 0 < Ki:
                    raise ValueError(f'There is a z- Neumann condition at K = {ind[i][4]} crossing a domain slice at K = {Ki}. Consider changing p_col.')

        return ind, n, vd
