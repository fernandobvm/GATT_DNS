class Mesh:
    def __init__(self, nx, ny, nz, X, Y, Z=None, trackedNorm=False):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.X = X
        self.Y = Y
        self.Z = Z
        self.trackedNorm = trackedNorm

class FlowParameters:
    def __init__(self, Re, Ma, Pr, T0, gamma):
        self.Re = Re
        self.Ma = Ma
        self.Pr = Pr
        self.T0 = T0
        self.gamma = gamma

class Time:
    def __init__(self, dt, maxCFL, nStep, control, qtimes, tmax, CFLignoreZ=False):
        self.dt = dt
        self.maxCFL = maxCFL
        self.nStep = nStep
        self.control = control
        self.qtimes = qtimes
        self.tmax = tmax
        self.CFLignoreZ = CFLignoreZ

class NumMethods:
    def __init__(self, timeStepping, SFD=None, spatialFilterTime=None):
        self.timeStepping = timeStepping
        self.SFD = SFD
        self.spatialFilterTime = spatialFilterTime

class SFDParams:
    def __init__(self, SFD_type, Delta, X, resume):
        self.type = SFD_type
        self.Delta = Delta
        self.X = X
        self.resume = resume

def write_fortran_parameters(case_name, mesh, flow_parameters, time, num_methods, log_all, p_row, p_col):
    with open(f'{case_name}/bin/parameters.F90', 'w') as out_file:
        # Mesh parameters
        out_file.write(f'    integer :: nx = {mesh.nx}\n')
        out_file.write(f'    integer :: ny = {mesh.ny}\n')
        out_file.write(f'    integer :: nz = {mesh.nz}\n\n')
        
        # Flow parameters
        out_file.write(f'    real*8 :: Re = {flow_parameters.Re:.20f}d0\n')
        out_file.write(f'    real*8 :: Ma = {flow_parameters.Ma:.20f}d0\n')
        out_file.write(f'    real*8 :: Pr = {flow_parameters.Pr:.20f}d0\n')
        out_file.write(f'    real*8 :: T0 = {flow_parameters.T0:.20f}d0\n')
        out_file.write(f'    real*8 :: gamma = {flow_parameters.gamma:.20f}d0\n\n')
        
        # Time parameters
        out_file.write(f'    real*8 :: dtmax = {time.dt:.20f}d0\n')
        out_file.write(f'    real*8 :: maxCFL = {time.maxCFL:.20f}d0\n')

        if log_all == 0:
            log_all = 2147483647
        out_file.write(f'    integer :: logAll = {log_all}\n')
        out_file.write(f'    integer :: nSave = {time.nStep}\n')

        if not mesh.trackedNorm:
            out_file.write(f'    real*8 :: trackedNorm = 0.d0\n')
        else:
            norm_val = 1 / ((flow_parameters.gamma ** 2 - flow_parameters.gamma) * flow_parameters.Ma ** 2)
            out_file.write(f'    real*8 :: trackedNorm = {norm_val:.20f}d0\n')

        # Time control
        if time.control == 'dt':
            out_file.write('    integer :: timeControl = 1\n')
            out_file.write(f'    integer :: qTimesInt = {time.qtimes}\n')
            out_file.write('    real*8  :: qTimesReal\n\n')
            out_file.write(f'    integer :: tmaxInt = {time.tmax}\n')
            out_file.write('    real*8  :: tmaxReal\n\n')
        elif time.control == 'cfl':
            out_file.write('    integer :: timeControl = 2\n')
            out_file.write('    integer :: qTimesInt\n')
            out_file.write(f'    real*8  :: qtimesReal = {time.qtimes:.20f}d0\n\n')
            out_file.write('    integer :: tmaxInt\n')
            out_file.write(f'    real*8  :: tmaxReal = {time.tmax:.20f}d0\n\n')
        else:
            raise ValueError('Unrecognized type of time control. Use either dt or cfl')

        # Time stepping methods
        if num_methods.timeStepping == 'RK4':
            out_file.write('    integer :: timeStepping = 1\n')
        elif num_methods.timeStepping == 'Euler':
            out_file.write('    integer :: timeStepping = 2\n')
        elif num_methods.timeStepping == 'SSPRK3':
            out_file.write('    integer :: timeStepping = 3\n')
        else:
            raise ValueError('Unrecognized time stepping method')

        # SFD parameters
        if num_methods.SFD:
            out_file.write(f'    integer :: SFD = {num_methods.SFD.type}\n')
            out_file.write(f'    real*8 :: SFD_Delta = {num_methods.SFD.Delta:.20f}d0\n')
            out_file.write(f'    real*8 :: SFD_X_val = {num_methods.SFD.X:.20f}d0\n')
            out_file.write(f'    integer :: resumeMeanFlow = {num_methods.SFD.resume}\n\n')
        else:
            out_file.write('    integer :: SFD = 0\n')
            out_file.write('    real*8 :: SFD_Delta = 0.d0\n')
            out_file.write('    real*8 :: SFD_X_val = 0.d0\n')
            out_file.write('    integer :: resumeMeanFlow = 0\n\n')

        # Spatial filter time
        if not num_methods.spatialFilterTime or num_methods.spatialFilterTime <= 0:
            out_file.write('    real*8 :: FilterCharTime = -1.d0\n')
        else:
            out_file.write(f'    real*8 :: FilterCharTime = {num_methods.spatialFilterTime:.20f}d0\n')

        # Mesh differences
        if mesh.nz == 1:
            dxmin = [1 / min(mesh.X[1:] - mesh.X[:-1]), 1 / min(mesh.Y[1:] - mesh.Y[:-1]), 0]
        else:
            dxmin = [1 / min(mesh.X[1:] - mesh.X[:-1]), 1 / min(mesh.Y[1:] - mesh.Y[:-1]), 1 / min(mesh.Z[1:] - mesh.Z[:-1])]

        if time.CFLignoreZ:
            dxmin[2] = 0

        out_file.write(f'    real*8, dimension(3) :: dxmin = (/ {dxmin[0]:.20f}d0, {dxmin[1]:.20f}d0, {dxmin[2]:.20f}d0 /)\n\n')

        # Parameters for p_row and p_col
        out_file.write(f'    integer :: p_row = {p_row}\n')
        out_file.write(f'    integer :: p_col = {p_col}\n')

