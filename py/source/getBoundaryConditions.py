#TODO: Verificar esse código com a versão não mini do gpt

class BoundaryConditions:
    def __init__(self, flow_type, mesh, flow_parameters, neumann_order):
        self.gamma = flow_parameters.gamma
        self.E0 = 1 / ((self.gamma**2 - self.gamma) * flow_parameters.Ma**2)
        self.P0 = (self.gamma - 1) * self.E0
        
        self.var = []
        self.type = []
        self.dir = []
        self.val = []
        self.xi = []
        self.xf = []
        self.yi = []
        self.yf = []
        self.zi = []
        self.zf = []
        
        if not self.load_boundary_file(flow_type.name):
            raise FileNotFoundError(f'Boundary condition file not found: {flow_type.name}')

        self.boundary = self.create_boundary_structure()
        self.calculate_wall_region(mesh)
        self.neumann_coeffs = self.get_neumann_coeffs(neumann_order)

    def load_boundary_file(self, name):
        # Implement the logic to load and execute the boundary file
        # For example, you can use exec to run the contents of the boundary file
        boundary_file_path = f'source/boundaries/{name}.py'
        try:
            with open(boundary_file_path) as f:
                exec(f.read())
            return True
        except FileNotFoundError:
            return False

    def create_boundary_structure(self):
        boundary = {
            'var': self.var,
            'type': self.type,
            'dir': self.dir,
            'val': self.val,
            'xi': self.xi,
            'xf': self.xf,
            'yi': self.yi,
            'yf': self.yf,
            'zi': self.zi,
            'zf': self.zf,
            'E0': self.E0,
            'gamma': self.gamma,
            # Initialize wall limits and corners here
            'wall': {
                'up': [],
                'down': [],
                'front': [],
                'back': [],
                'right': [],
                'left': [],
            },
            'corners': [],
            'insideWall': None,  # To be calculated later
            'disturb': []
        }
        return boundary

    def calculate_wall_region(self, mesh):
        self.boundary['insideWall'] = ~self.flowRegion  # Assuming flowRegion is defined elsewhere
        for wall_limits in [self.wallUpLimits, self.wallDownLimits, self.wallFrontLimits, 
                            self.wallBackLimits, self.wallRightLimits, self.wallLeftLimits]:
            for limit in wall_limits:
                self.boundary['insideWall'][limit[0]:limit[1], limit[2]:limit[3], limit[4]:limit[5]] = False

    def get_neumann_coeffs(self, neumann_order):
        NC1 = [[-1, 1], [-3/2, 2, -1/2], [-11/6, 3, -3/2, 1/3], 
                [-25/12, 4, -3, 4/3, -1/4], [-137/60, 5, -5, 10/3, -5/4, 1/5], 
                [-49/20, 6, -15/2, 20/3, -15/4, 6/5, -1/6]]
        
        NC2 = [[1, -2, 1], [2, -5, 4, -1], [35/12, -26/3, 19/2, -14/3, 11/12], 
                [15/4, -77/6, 107/6, -13, 61/12, -5/6], 
                [203/45, -87/5, 117/4, -254/9, 33/2, -27/5, 137/180], 
                [469/90, -223/10, 879/20, -949/18, 41, -201/10, 1019/180, -7/10]]
        
        return {
            'neumannCoeffs': -NC1[neumann_order[0]][1:] / NC1[neumann_order[0]][0],
            'neumann2Coeffs': -NC2[neumann_order[1]][1:] / NC2[neumann_order[1]][0],
        }

    def add_disturbances(self, flow_type):
        if hasattr(flow_type, 'disturb'):
            for disturbance in flow_type.disturb:
                if disturbance.active:
                    self.boundary['disturb'].append({
                        'type': disturbance.type,
                        'forcing': hasattr(disturbance, 'forcing') and disturbance.forcing,
                        'par': getattr(disturbance, 'par', []),
                        'var': disturbance.var,
                        'ind': self.calculate_indices(disturbance, mesh)
                    })

    def calculate_indices(self, disturbance, mesh):
        xi = disturbance.x[0]
        xf = disturbance.x[1]
        yi = disturbance.y[0]
        yf = disturbance.y[1]
        zi = disturbance.z[0]
        zf = disturbance.z[1]
        
        # Replace inf with mesh boundaries
        xi = self.replace_inf_with_mesh_bounds(xi, mesh.X)
        xf = self.replace_inf_with_mesh_bounds(xf, mesh.X)
        yi = self.replace_inf_with_mesh_bounds(yi, mesh.Y)
        yf = self.replace_inf_with_mesh_bounds(yf, mesh.Y)
        zi = self.replace_inf_with_mesh_bounds(zi, mesh.Z)
        zf = self.replace_inf_with_mesh_bounds(zf, mesh.Z)
        
        return [
            self.find_index(mesh.X, xi),
            self.find_index(mesh.X, xf),
            self.find_index(mesh.Y, yi),
            self.find_index(mesh.Y, yf),
            self.find_index(mesh.Z, zi),
            self.find_index(mesh.Z, zf),
        ]

    def replace_inf_with_mesh_bounds(self, value, mesh_array):
        if value == float('inf'):
            return mesh_array[-1]
        elif value == -float('inf'):
            return mesh_array[0]
        return value

    def find_index(self, array, value):
        return (np.abs(array - value)).argmin()

# Uso da classe
# boundary, mesh = BoundaryConditions(flow_type, mesh, flow_parameters, neumann_order)
