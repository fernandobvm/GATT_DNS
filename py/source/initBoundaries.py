import numpy as np

class Boundary:
    def __init__(self, boundary_data, mesh, domain_slices_y, domain_slices_z, p_row, p_col):
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
        
        self.init_boundaries(boundary_data, mesh, domain_slices_y, domain_slices_z, p_row, p_col)

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

    def init_boundaries(self, boundary_data, mesh, domain_slices_y, domain_slices_z, p_row, p_col):
        direction_order = ['xi', 'xf', 'yi', 'yf', 'zi', 'zf']
        
        for i in range(len(boundary_data['val'])):
            if boundary_data['type'][i] == 'dir':
                self.process_dir_boundary(boundary_data, i)
            elif boundary_data['type'][i] == 'neu':
                self.process_neu_boundary(boundary_data, i, direction_order)
            elif boundary_data['type'][i] == 'sec':
                self.process_sec_boundary(boundary_data, i, direction_order)

        self.split_boundaries(mesh, domain_slices_y, domain_slices_z, p_row, p_col)

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

    def split_boundaries(self, mesh, domain_slices_y, domain_slices_z, p_row, p_col):
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
                for i in range(len(boundary['disturb'])):
                    if boundary['disturb'][i]:
                        ind = boundary['disturb'][i]['ind']
                        ind[2:6] = [
                            max(ind[2], domain_slices_y[0][j]),
                            min(ind[3], domain_slices_y[1][j]),
                            max(ind[4], domain_slices_z[0][k]),
                            min(ind[5], domain_slices_z[1][k])
                        ]

                        if ind[2] <= ind[3] and ind[4] <= ind[5]:
                            disturb.append(boundary['disturb'][i])
                            disturb[-1]['ind'] = ind
                            disturb[-1]['X'] = mesh['X'][ind[0]:ind[1]]
                            disturb[-1]['Y'] = mesh['Y'][ind[2]:ind[3]]
                            disturb[-1]['Z'] = mesh['Z'][ind[4]:ind[5]]

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
