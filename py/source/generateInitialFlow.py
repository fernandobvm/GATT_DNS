import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import odeint

class InitialFlowGenerator:
    def __init__(self, mesh, flow_parameters, initial_flow, walls, flow_name):
        self.mesh = mesh
        self.flow_parameters = flow_parameters
        self.initial_flow = initial_flow
        self.walls = walls
        self.flow_name = flow_name
        self.gamma = flow_parameters.gamma
        self.Ma = flow_parameters.Ma
        self.Re = flow_parameters.Re
        self.E0 = 1 / ((self.gamma**2 - self.gamma) * self.Ma**2)

        # Atributos para armazenar os resultados
        self.U = None
        self.V = None
        self.W = None
        self.R = None
        self.E = None

        self.generate_initial_flow()

    def generate_initial_flow(self):
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        initial_type = self.initial_flow.type

        if initial_type == 'uniform':
            self.uniform_flow(nx, ny, nz)

        elif initial_type == 'poiseuille':
            self.poiseuille_flow(nx, ny, nz)

        elif initial_type == 'blasius':
            self.blasius_flow(nx, ny)

        elif initial_type in ['compressibleBL_isothermal', 'compressibleBL_adiabatic']:
            self.compressible_blow_flow(initial_type)

        elif initial_type == 'file':
            self.file_flow(nx, ny, nz)

        else:
            raise ValueError(f"Initial flow type not implemented: {initial_type}")

    def uniform_flow(self, nx, ny, nz):
        self.U = np.ones((nx, ny, nz))
        self.V = np.zeros((nx, ny, nz))
        self.W = np.zeros((nx, ny, nz))
        self.R = np.ones((nx, ny, nz))
        self.E = np.ones((nx, ny, nz)) * self.E0

        y0ind = np.argmin(np.abs(self.mesh.Y))

        if 'boundaryLayer' in self.flow_name:
            self.U[:, :y0ind, :] = 0

        if hasattr(self.initial_flow, 'U0'):
            self.U *= self.initial_flow.U0

    def poiseuille_flow(self, nx, ny, nz):
        self.U = np.zeros((nx, ny, nz))
        self.V = np.zeros((nx, ny, nz))
        self.W = np.zeros((nx, ny, nz))
        self.R = np.ones((nx, ny, nz))
        self.E = np.ones((nx, ny, nz)) * self.E0

        eta = (self.mesh.Y - self.mesh.Y[0]) / (self.mesh.Y[-1] - self.mesh.Y[0])
        u0 = self.flow_parameters.U0
        u1 = self.flow_parameters.lowerWallVelocity
        u2 = self.flow_parameters.upperWallVelocity

        self.U += (-6 * u0 + 3 * u1 + 3 * u2) * eta**2 + (6 * u0 - 4 * u1 - 2 * u2) * eta + u1

    def blasius_flow(self, nx, ny):
        ybl = np.arange(0, 10, 0.0001)
        ubl = self.blasius(ybl)
        thetabl = 0.664155332943009
        self.R = np.ones((nx, ny))
        self.U = np.ones((nx, ny))
        self.V = np.zeros((nx, ny))
        self.W = np.zeros((nx, ny))
        self.E = np.ones((nx, ny)) * self.E0

        Y0 = np.zeros(nx)
        if hasattr(self.initial_flow, 'blasiusFit'):
            walls2D = np.any(self.walls, axis=2)
            for i in range(nx):
                Y0[i] = self.mesh.Y[np.where(walls2D[i] == 0)[0][0]]

        for i in range(nx):
            xi = self.mesh.X[i]
            if xi > 0:
                theta = 0.664 * np.sqrt(xi / self.Re)
                self.U[i, :] = interp1d(ybl * theta / thetabl + Y0[i], ubl, fill_value="extrapolate")(self.mesh.Y)

        self.U[self.U > 1] = 1
        self.U[np.isnan(self.U)] = 1
        self.U = np.repeat(self.U[:, :, np.newaxis], self.mesh.nz, axis=2)

    def compressible_blow_flow(self, flow_type):
        compressibleBL_flow = self.calc_compressible_BL(flow_type)
        self.U = compressibleBL_flow.U
        self.V = compressibleBL_flow.V
        self.W = compressibleBL_flow.W
        self.E = compressibleBL_flow.E
        self.R = compressibleBL_flow.R

    def file_flow(self, nx, ny, nz):
        flow_file = np.load(self.initial_flow.flowFile, allow_pickle=True).item()
        self.U = flow_file['U']
        self.V = flow_file['V']
        self.W = flow_file['W']
        self.R = flow_file['R']
        self.E = flow_file['E']

        if hasattr(self.initial_flow, 'meshFile'):
            mesh_file = np.load(self.initial_flow.meshFile, allow_pickle=True).item()
            Xfile = mesh_file['X']
            Yfile = mesh_file['Y']
            Zfile = mesh_file['Z']

            self.U = self.interpolate_flow(self.U, Xfile, Yfile, Zfile, nx, ny, nz)

            if hasattr(self.initial_flow, 'changeMach') and self.initial_flow.changeMach:
                if mesh_file['flowParameters'].Ma != self.Ma:
                    print(f"Initial flow file has a Mach number of {mesh_file['flowParameters'].Ma} which will be changed to {self.Ma} to match the current simulation.")
                    self.E *= mesh_file['flowParameters'].Ma**2 / self.Ma**2

    def interpolate_flow(self, Ufile, Xfile, Yfile, Zfile, nx, ny, nz):
        Xmesh, Ymesh, Zmesh = np.meshgrid(self.mesh.X, self.mesh.Y, self.mesh.Z, indexing='ij')
        Ufile = np.nan_to_num(Ufile, nan=0)

        if nz == 1 or len(Zfile) == 1:  # 2D case
            interp_func_U = interp1d((Xfile.flatten(), Yfile.flatten()), Ufile.flatten(), bounds_error=False, fill_value=0)
            return interp_func_U((Xmesh.flatten(), Ymesh.flatten())).reshape(Xmesh.shape)
        else:  # 3D case
            interp_func_U = RegularGridInterpolator((Xfile, Yfile, Zfile), Ufile, bounds_error=False, fill_value=0)
            return interp_func_U((Xmesh, Ymesh, Zmesh))

    def add_noise(self):
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        noise_type = getattr(self.initial_flow, 'noiseType', 'rand')
        add_noise = getattr(self.initial_flow, 'addNoise', 0)

        noiseU = add_noise * np.random.randn(nx, ny, nz)
        noiseV = add_noise * np.random.randn(nx, ny, nz)
        noiseW = add_noise * np.random.randn(nx, ny, nz)
        noiseR = add_noise * np.random.randn(nx, ny, nz)
        noiseE = add_noise * np.random.randn(nx, ny, nz)

        if hasattr(self.initial_flow, 'noiseCenter'):
            x0, y0, z0 = self.initial_flow.noiseCenter
            sigmaX = getattr(self.initial_flow, 'noiseSigma', np.inf)
            sigmaY = getattr(self.initial_flow, 'noiseSigma', np.inf)
            sigmaZ = getattr(self.initial_flow, 'noiseSigma', np.inf)

        else:
            x0, y0, z0 = 0, 0, 0
            sigmaX, sigmaY, sigmaZ = np.inf, np.inf, np.inf

        radius = ((self.mesh.X[:, np.newaxis] - x0) ** 2 / sigmaX +
                   (self.mesh.Y[np.newaxis, :] - y0) ** 2 / sigmaY +
                   (self.mesh.Z[np.newaxis, np.newaxis, :] - z0) ** 2 / sigmaZ)

        noise_multiplier = np.exp(-radius)

        self.U += noiseU * noise_multiplier
        self.V += noiseV * noise_multiplier
        self.W += noiseW * noise_multiplier
        self.R += noiseR * noise_multiplier
        self.E += noiseE * noise_multiplier

    def blasius(self, y):
        def blasius_eq(f, x):
            d = np.zeros(3)
            d[0] = f[1]
            d[1] = f[2]
            d[2] = -0.5 * f[0] * f[2]
            return d

        f0 = [0, 0, 0.33204312]
        result = odeint(blasius_eq, f0, y)
        return result[:, 1]
