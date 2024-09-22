import numpy as np

class Mesh:
    def __init__(self):
        self.x = MeshAxis()
        self.y = MeshAxis()
        self.z = MeshAxis()
        self.tracked_points = []
        self.tracked_norm = True

    def generateMesh(self, domain):
        #TODO: implementar este método, ele é utilizado no preprocessing.m
        self.nx = 0  # number of grid points in x
        self.ny = 0  # number of grid points in y
        self.nz = 0  # number of grid points in z
        self.X = 0    # X coordinates of the mesh
        self.Y = 0    # Y coordinates of the mesh
        self.Z = 0    # Z coordinates of the mesh
        #extraRefinement é um objeto do meshAxis
        #trackedNorm = tracked_norm
        #trackedPoint = tracked_points
#class MeshAxis:
#    def __init__(self, axisType, axisValues):
#        self.axisType = axisType  # Type of axis (e.g., 'X', 'Y', 'Z')
#        self.axisValues = axisValues  # Values along the axis
class MeshAxis:
    def __init__(self):
        self.type = 'attractors'
        self.attractor_points = []
        self.attractor_strength = []
        self.attractor_size = []
        self.attractor_regions = []
        self.match_fixed = 2
        self.periodic = False
        self.fix_periodic_domain_size = False
        self.extra_refinement = 0
        self.d0 = None
        self.buffer_i = Buffer()  # Buffer for initial part
        self.buffer_f = Buffer()  # Buffer for final part


class FlowParameters:
    def __init__(self, Re, Ma, Pr=0.71, gamma=1.4, T0=300):
        self.Re = Re        # Reynolds number
        self.Ma = Ma        # Mach number
        self.Pr = Pr        # Prandtl number
        self.T0 = T0        # Temperature
        self.gamma = gamma  # Ratio of specific heats

class Domain:
    def __init__(self, xi, xf, yi, yf, zi, zf):
        self.xi = xi
        self.xf = xf
        self.yi = yi
        self.yf = yf
        self.zi = zi
        self.zf = zf

class Time:
    def __init__(self, dt, max_cfl,  control, qtimes, tmax, nStep = None, CFLignoreZ=False):
        self.dt = dt                # time step size
        self.tmax = tmax            # max time value
        self.nStep = nStep          # number of steps for saving data | usado no checkPreviousRun.m
        self.max_cfl = max_cfl      # max CFL number
        self.qtimes = qtimes        # query times for controlling time
        self.control = control      # control type ('dt' or 'cfl')
        self.CFLignoreZ = CFLignoreZ  # flag to ignore CFL condition in Z direction


class NumericalMethods:
    def __init__(self, spatial_derivs = 'SL6', spatial_derivs_buffer = 'EX4', time_stepping = 'RK4', neumann_order = 6, neumann2_order = 2, spatial_filter_strength = 0.49, spatial_filter_time = 0, filter_directions = [1, 1, 1], filter_borders = 'reducedOrder', filter_borders_start_x = False, filter_borders_end_x = False, filter_borders_end_y = False, sfd = None):
        self.neumann_order = neumann_order
        self.time_stepping = time_stepping
        self.spatial_derivs = spatial_derivs
        self.neumann2_order = neumann2_order
        self.spatial_filter_time = spatial_filter_time
        self.spatial_derivs_buffer = spatial_derivs_buffer
        self.spatial_filter_strength = spatial_filter_strength
        self.filter_directions = filter_directions
        self.filter_borders = filter_borders
        self.filter_borders_start_x = filter_borders_start_x
        self.filter_borders_end_x = filter_borders_end_x
        self.filter_borders_end_y = filter_borders_end_y
        if sfd is None:
            self.SFD = SFD()
        else:
            self.SFD = sfd


class SFD:
    def __init__(self, sfd_type=2, X=0.05, Delta=10, applyY=False):
        self.X = X
        self.Delta = Delta
        self.type = sfd_type
        self.applyY = applyY
        self.extra_region = []
    
    def add_extra_region(self, location, size, X):
        self.extra_region.append({"location": location, "size": size, "X": X})

class Matrices:
    def __init__(self, x, y, z, neumannCoeffs, neumann2Coeffs):
        self.x = x  # matrix block for x direction
        self.y = y  # matrix block for y direction
        self.z = z  # matrix block for z direction (only if K > 1)
        self.neumannCoeffs = neumannCoeffs  # Neumann boundary coefficients
        self.neumann2Coeffs = neumann2Coeffs  # Second Neumann coefficients


class MatrixDirection:
    def __init__(self, blocks, nRHS, nRHSf, A, B, C, R, Af, Bf, Cf, Rf, D=None, Df=None, periodic=False, nTypes=0):
        self.blocks = blocks  # matrix blocks
        self.nRHS = nRHS  # number of RHS terms for derivatives
        self.nRHSf = nRHSf  # number of RHS terms for filters
        self.A = A  # matrix A
        self.B = B  # matrix B
        self.C = C  # matrix C
        self.R = R  # matrix R (RHS)
        self.Af = Af  # filter matrix A
        self.Bf = Bf  # filter matrix B
        self.Cf = Cf  # filter matrix C
        self.Rf = Rf  # filter matrix R
        self.D = D  # optional: matrix D for periodic cases
        self.Df = Df  # optional: filter matrix D for periodic cases
        self.periodic = periodic  # periodic flag
        self.nTypes = nTypes  # number of matrix types

class Disturbance:
    def __init__(self, x_range, y_range, z_range, var, disturb_type, par, active=False, fit_points=False):
        self.x = x_range
        self.y = y_range
        self.z = z_range
        self.var = var
        self.type = disturb_type
        self.extraNodes = np.zeros(6)
        self.par = par
        self.active = active
        self.fitPoints = fit_points
class Disturbances:
    def __init__(self, disturbanceType, amplitude, frequency, phase):
        self.disturbanceType = disturbanceType  # Type of disturbance (e.g., acoustic, entropy, etc.)
        self.amplitude = amplitude  # Amplitude of the disturbance
        self.frequency = frequency  # Frequency of the disturbance
        self.phase = phase  # Phase shift of the disturbance

class Cavity:
    def __init__(self, x_range, y_range, z_range, cavityDepth = None, cavityWidth = None, cavityLocation = None):
        self.x = x_range
        self.y = y_range
        self.z = z_range
        self.cavityDepth = cavityDepth  # Depth of the cavity
        self.cavityWidth = cavityWidth  # Width of the cavity
        self.cavityLocation = cavityLocation  # Location of the cavity in the domain

class FlowType:
    def __init__(self, flowCondition = None, flowRegime = None):
        self.name = 'boundaryLayerIsothermal'
        self.initial_type = 'blasius'
        self.cav = []
        self.disturb = []
        self.flowCondition = flowCondition  # Condition of the flow (e.g., subsonic, supersonic)
        self.flowRegime = flowRegime  # Regime of the flow (laminar, turbulent, etc.)
        
class Buffer:
    def __init__(self, n=0, buffer_type='sigmoid', stretching=0, transition=0.2, ramp=None, bufferSize = None, bufferType = None):
        self.n = n
        self.type = buffer_type
        self.stretching = stretching
        self.transition = transition
        self.ramp = ramp
        self.bufferSize = bufferSize  # Size of the buffer zone
        self.bufferType = bufferType  # Type of buffer zone (e.g., sponge, absorbing)
    
    def apply_buffer(self):
        # Implement the logic to apply the buffer to the mesh
        pass

class MeshBuffer:
    def __init__(self, n, buffer_type, stretching, transition):
        self.n = n
        self.type = buffer_type
        self.stretching = stretching
        self.transition = transition
class MeshBuffer:
    def __init__(self, mesh, buffer):
        self.mesh = mesh  # Mesh object
        self.buffer = buffer  # Buffer object associated with the mesh





# Helper function for creating empty mesh and flow parameter classes in other files
def create_empty_classes():
    return Mesh(0, 0, 0), FlowParameters(0, 0, 0, 0, 0), Time(0, 0, 0, '', 0, 0), NumericalMethods('', [0, 0, 0]), Matrices(None, None, None, [], [])
