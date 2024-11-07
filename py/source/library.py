import os
import re
import copy
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator, PchipInterpolator, CubicSpline
from scipy.integrate import odeint, simpson 

########## CLASSES

class Mesh:
    def __init__(self):
        self.x = MeshAxis()
        self.y = MeshAxis()
        self.z = MeshAxis()
        self.tracked_points = []
        self.tracked_norm = True
        self.fit_tracked_points = False
        self.nx = 0  # number of grid points in x
        self.ny = 0  # number of grid points in y
        self.nz = 0  # number of grid points in z
        self.X = 0    # X coordinates of the mesh
        self.Y = 0    # Y coordinates of the mesh
        self.Z = 0    # Z coordinates of the mesh

    def generateMesh(self, domain):
        #TODO: implementar este método, ele é utilizado no preprocessing.m
        self.nx = 0  # number of grid points in x
        self.ny = 0  # number of grid points in y
        self.nz = 0  # number of grid points in z
        self.X = 0    # X coordinates of the mesh
        self.Y = 0    # Y coordinates of the mesh
        self.Z = 0    # Z coordinates of the mesh
        #extra_refinement é um objeto do meshAxis
        #trackedNorm = tracked_norm
        #trackedPoint = tracked_points
    def generate_mesh(self, xi, xf, direction):
        # Gerar malha arbitrária se o número de nós não for definido
        mesh_dir = getattr(self, direction.lower())
        d0 = getattr(mesh_dir, "d0")
        if mesh_dir.n is None:
            #mesh_temp = Mesh()
            mesh_temp = copy.deepcopy(self)
            
            mesh_temp2 = getattr(mesh_temp, direction.lower())
            mesh_temp2.n = np.ceil((xf - xi) / d0).astype(int)
            mesh_temp2.match_fixed = 0
            mesh_temp2.extra_refinement = 0
            mesh_temp2.fix_periodic_domain_size = 0
            mesh_temp2.buffer_i.n = 0
            mesh_temp2.buffer_f.n = 0
            Xtemp, _ = mesh_temp.generate_mesh(xi, xf, direction)

            d_base = np.max(np.diff(Xtemp))
            mesh_dir.n = np.ceil(mesh_temp2.n * d_base / d0).astype(int)

        # Definir número de nós para zonas tampão
        if mesh_dir.buffer_i.n is None or mesh_dir.buffer_f.n is None:
            mesh_temp = copy.deepcopy(self)
            mesh_temp2 = getattr(mesh_temp, direction.lower())
            mesh_temp2.match_fixed = 0
            mesh_temp2.extra_refinement = 0
            mesh_temp2.fix_periodic_domain_size = 0
            mesh_temp2.buffer_i.n = 0
            mesh_temp2.buffer_f.n = 0
            Xtemp, _ = mesh_temp.generate_mesh(xi, xf, direction)

            if mesh_dir.buffer_i.n is None:
                d_base = Xtemp[1] - Xtemp[0]
                target = mesh_dir.buffer_i['l']
                mesh_dir.buffer_i.n = np.floor(target / d_base).astype(int)
                self._adjust_buffer_zone('i', target, d_base)

            if mesh_dir.buffer_f.n is None:
                d_base = Xtemp[-1] - Xtemp[-2]
                target = mesh_dir.buffer_f['l']
                mesh_dir.buffer_f.n = np.floor(target / d_base).astype(int)
                self._adjust_buffer_zone('f', target, d_base)

        # Se a malha for um único ponto
        if mesh_dir.n == 1:
            mesh_dir.type = 'uniform'

        # Remover zona tampão para dimensões periódicas
        if mesh_dir.periodic and (mesh_dir.buffer_i.n > 0 or mesh_dir.buffer_f.n > 0):
            print(f"Warning: Buffer zone was removed for periodic dimension {direction}")
            mesh_dir.buffer_i.n = 0
            mesh_dir.buffer_f.n = 0

        # Adicionar nó temporário se necessário
        added_temp_node = False
        if mesh_dir.periodic and mesh_dir.fix_periodic_domain_size and mesh_dir.n > 1 and mesh_dir.type != 'file':
            mesh_dir.n += 1
            added_temp_node = True

        # Contar nós
        nx = mesh_dir.n + mesh_dir.buffer_i.n + mesh_dir.buffer_f.n
        physical_start = mesh_dir.buffer_i.n
        physical_end = mesh_dir.buffer_i.n + mesh_dir.n

        # Computar domínio físico
        X_physical = self._compute_physical_domain(mesh_dir, xi, xf)

        # Ajustar pontos fixos como cavidades
        if mesh_dir.match_fixed and mesh_dir.n > 1:
            X_physical = self._match_fixed_points(mesh_dir, X_physical)

        X = np.zeros(nx)
        X[physical_start:physical_end] = X_physical

        # Adicionar zonas tampão
        if mesh_dir.buffer_i.n > 0:
            base_dist = X[physical_start + 1] - X[physical_start]
            XB = self._calc_buffer_zone(mesh_dir.buffer_i) * base_dist
            X[:physical_start] = X[physical_start] - XB[::-1]

        if mesh_dir.buffer_f.n > 0:
            base_dist = X[physical_end - 1] - X[physical_end - 2]
            XB = self._calc_buffer_zone(mesh_dir.buffer_f) * base_dist
            X[physical_end:] = X[physical_end - 1] + XB

        # Adicionar refinamento extra
        if mesh_dir.extra_refinement > 0:
            X = self._add_extra_refinement(mesh_dir, X)

        # Remover nó temporário
        if added_temp_node:
            X = X[:-1]
            mesh_dir.n -= 1
            nx -= 1

        return X, nx

    def _compute_physical_domain(self, mesh_dir, xi, xf):
        if mesh_dir.type == 'uniform':
            XPhysical = custom_linspace(xi, xf, mesh_dir.n)
        
        elif mesh_dir.type == 'power':
            XPhysical = xi + (xf - xi) * custom_linspace(0, 1, mesh_dir.n) ** mesh_dir.power
        
        elif mesh_dir.type == 'tanh':
            if mesh_dir.local == 'f':  # Refine at the end
                eta = np.tanh(custom_linspace(0, mesh_dir.par, mesh_dir.n))
                eta /= eta[-1]
            
            elif mesh_dir.local == 'i':  # Refine at the start
                eta = np.tanh(custom_linspace(0, mesh_dir.par, mesh_dir.n))
                eta /= eta[-1]
                eta = 1 - eta[::-1]
            
            elif mesh_dir.local == 'b':  # Refine at both sides
                eta = np.tanh(custom_linspace(-mesh_dir.par, mesh_dir.par, mesh_dir.n))
                eta /= eta[-1]
                eta = (eta + 1) / 2
            
            XPhysical = (xf - xi) * eta + xi
        
        elif mesh_dir.type == 'attractors':
            xBase = custom_linspace(xi, xf, 100 * mesh_dir.n)
            eta = np.ones(100 * mesh_dir.n)
            
            # Apply attractor points
            for i in range(len(mesh_dir.attractor_points)):
                eta += mesh_dir.attractor_strength[i] * np.exp(-((xBase - mesh_dir.attractor_points[i]) / mesh_dir.attractor_size[i]) ** 2)
            
            # Apply attractor regions
            if hasattr(mesh_dir, 'attractor_regions'):
                if len(mesh_dir.attractor_regions.shape) == 1:
                    for i in range(len(mesh_dir.attractor_regions.shape)):
                        nodePositions = mesh_dir.attractor_regions[0:4].copy()
                        if np.isinf(nodePositions[1]):
                            nodePositions[0:2] = [xi - 2, xi - 1]
                        if np.isinf(nodePositions[2]):
                            nodePositions[2:4] = [xf + 1, xf + 2]
                        nodePositions = np.array([nodePositions[0] - 1] + list(nodePositions) + [nodePositions[3] + 1])
                        interp = PchipInterpolator(nodePositions, [0, 0, 1, 1, 0, 0])

                        eta += mesh_dir.attractor_regions[4] * interp(xBase) ** 2
                else:                    
                    for i in range(mesh_dir.attractor_regions.shape[0]):
                        nodePositions = mesh_dir.attractor_regions[i][0:4].copy()
                    
                        if np.isinf(nodePositions[1]):
                            nodePositions[0:2] = [xi - 2, xi - 1]
                        if np.isinf(nodePositions[2]):
                            nodePositions[2:4] = [xf + 1, xf + 2]
                    
                        nodePositions = np.array([nodePositions[0] - 1] + list(nodePositions) + [nodePositions[3] + 1])
                        interp = PchipInterpolator(nodePositions, [0, 0, 1, 1, 0, 0])
                        eta += mesh_dir.attractor_regions[i, 4] * interp(xBase) ** 2
            
            eta = np.cumsum(eta)
            eta = eta - eta[0]
            eta = (mesh_dir.n - 1) * eta / eta[-1] + 1
            eta[-1] = mesh_dir.n
            
            # Interpolate to find physical nodes
            #XPhysical = interp1d(eta, xBase, kind='spline')(np.arange(1, mesh_dir.n + 1))
            XPhysical = CubicSpline(eta, xBase)(np.arange(1, mesh_dir.n + 1))
        
        elif mesh_dir.type == 'attractors_old':
            xBase = custom_linspace(xi, xf, mesh_dir.n)
            eta = np.ones(mesh_dir.n)
            
            # Apply attractor points
            for i in range(len(mesh_dir.attractor_points)):
                eta += mesh_dir.attractor_strength[i] * np.exp(-((xBase - mesh_dir.attractor_points[i]) / mesh_dir.attractor_size[i]) ** 2)
            
            eta = np.cumsum(eta)
            eta = eta - eta[0]
            eta = (mesh_dir.n - 1) * eta / eta[-1] + 1
            eta[-1] = mesh_dir.n
            
            # Interpolate to find physical nodes
            #XPhysical = interp1d(eta, xBase, kind='spline')(np.arange(1, mesh_dir.n + 1))
            XPhysical = CubicSpline(eta, xBase)(np.arange(1, mesh_dir.n + 1))

        
        elif mesh_dir.type == 'file':
            X = np.load(mesh_dir.file)  # Load .npy file directly
            
            if len(X.shape) == 1:
                X = X.reshape(1, -1)  # Ensure correct shape
            
            # Check if file has the correct number of nodes
            if not hasattr(mesh_dir, 'fileCalcBuffer') or not mesh_dir.fileCalcBuffer:
                if X.shape[1] != mesh_dir.n:
                    raise ValueError(f"{mesh_dir.direction} mesh from file {mesh_dir.file} contains {X.shape[1]} nodes instead of {mesh_dir.n} as specified in parameters")
            XPhysical = X.flatten()
        
        return XPhysical

    def _adjust_buffer_zone(self, zone, target, d_base):
        #zone_data = self.buffer[zone]
        zone_data = getattr(self, 'buffer_' + zone)
        XB = self._calc_buffer_zone(zone_data)
        value = XB[-1] * d_base
        while value > target and zone_data.n > 2:
            zone_data.n = np.floor(zone_data.n / 2).astype(int)
            XB = self._calc_buffer_zone(zone_data)
            value = XB[-1] * d_base
        while value < target:
            zone_data.n += 1
            XB = self._calc_buffer_zone(zone_data)
            value = XB[-1] * d_base

    def _calc_buffer_zone(self, par):
        if par.type == 'exponential':
            if not hasattr(par, 'ramp'):
                XB = np.zeros(par.n)
                XB[0] = 1 + par.stretching
                for i in range(1, par.n):
                    XB[i] = XB[i - 1] + (1 + par.stretching) ** i
            else:
                delta = np.ones(par.n)
                for i in range(1, par.n):
                    stretching = 1 + min(1, i / par.ramp) * par.stretching
                    delta[i] = delta[i - 1] * stretching
                XB = np.cumsum(delta)
        elif par.type == 'sigmoid':
            xb = -10 + 12 * np.arange(par.n) / (par.n - 1)
            delta = 1 / (1 + np.exp(-xb))
            delta = delta * par.stretching + 1
            XB = np.cumsum(delta)
        return XB

    def _match_fixed_points(self, mesh_dir, X_physical):
        fix_points = np.unique([X_physical[0], X_physical[1]] + mesh_dir.fixPoints + [X_physical[-2], X_physical[-1]])
        closest_nodes = np.argmin(np.abs(np.subtract.outer(X_physical, fix_points)), axis=0)
        duplicates = np.where(np.diff(closest_nodes) == 0)[0]

        iter_max = mesh_dir.n
        iter_count = 0
        while len(duplicates) > 0 and iter_count <= iter_max:
            closest_nodes[duplicates] -= 1
            duplicates = np.where(np.diff(closest_nodes) == 0)[0]
            iter_count += 1

        if mesh_dir.match_fixed == 2:
            X_physical -= PchipInterpolator(closest_nodes, X_physical[closest_nodes] - fix_points)(np.arange(mesh_dir.n))
        else:
            #X_physical -= interp1d(closest_nodes, X_physical[closest_nodes] - fix_points, kind='spline')(np.arange(mesh_dir.n))
            X_physical -= CubicSpline(closest_nodes, X_physical[closest_nodes] - fix_points)(np.arange(mesh_dir.n))

        X_physical[closest_nodes] = fix_points
        return X_physical

    def _add_extra_refinement(self, mesh_dir, X):
        er = mesh_dir.extra_refinement
        self.n = (1 + er) * mesh_dir.n - er
        mesh_dir.buffer_i.n = (1 + er) * mesh_dir.buffer_i.n
        mesh_dir.buffer_f.n = (1 + er) * mesh_dir.buffer_f.n
        nx = mesh_dir.n + mesh_dir.buffer_i.n + mesh_dir.buffer_f.n
        Xfiner = np.zeros(nx)
        ix = 1
        for i in range(1, len(X)):
            Xfiner[ix - 1] = X[i - 1]
            for j in range(er):
                Xfiner[ix] = Xfiner[ix - 1] + (X[i] - X[i - 1]) / (er + 1)
                ix += 1
        Xfiner[-1] = X[-1]
        return Xfiner

    #meshAddFixedPoints
    def add_fixed_points(self, flow_type, domain):
        # Inicializar os fixPoints como em MATLAB
        self.x.fixPoints = [0]
        self.y.fixPoints = [0]
        self.z.fixPoints = []

        # Verificar se cavidades existem e adicionar pontos fixos
        if hasattr(flow_type, 'cav'):
            for cav in flow_type.cav:
                self.x.fixPoints.extend(cav.x)
                self.y.fixPoints.extend(cav.y)
                self.z.fixPoints.extend(cav.z)

        # Verificar se rugosidades existem e adicionar pontos fixos
        if hasattr(flow_type, 'rug'):
            for rug in flow_type.rug:
                self.x.fixPoints.extend(rug.x)
                self.y.fixPoints.extend(rug.y)
                self.z.fixPoints.extend(rug.z)

        # Verificar se distúrbios existem e adicionar pontos fixos
        if hasattr(flow_type, 'disturb'):
            for disturb in flow_type.disturb:
                if hasattr(disturb, 'fitPoints') and disturb.fitPoints:
                    self.x.fixPoints.extend(disturb.x)
                    self.y.fixPoints.extend(disturb.y)
                    self.z.fixPoints.extend(disturb.z)
                else:
                    if hasattr(disturb, 'fitPointsX') and disturb.fitPointsX:
                        self.x.fixPoints.extend(disturb.x)
                    if hasattr(disturb, 'fitPointsY') and disturb.fitPointsY:
                        self.y.fixPoints.extend(disturb.y)
                    if hasattr(disturb, 'fitPointsZ') and disturb.fitPointsZ:
                        self.z.fixPoints.extend(disturb.z)

        # Verificar se trackedPoints estão definidos e ajustar seus pontos
        if hasattr(self, 'tracked_points') and self.fit_tracked_points:
            for tracked_point in self.tracked_points:
                self.x.fixPoints.append(tracked_point[0])
                self.y.fixPoints.append(tracked_point[1])
                self.z.fixPoints.append(tracked_point[2])

        # Remover duplicados nos fixPoints
        self.x.fixPoints = list(np.unique(self.x.fixPoints))
        self.y.fixPoints = list(np.unique(self.y.fixPoints))
        self.z.fixPoints = list(np.unique(self.z.fixPoints))

        # Remover pontos fora do domínio ou com valor infinito
        self.x.fixPoints = [p for p in self.x.fixPoints if not np.isinf(p) and domain.xi <= p <= domain.xf]
        self.y.fixPoints = [p for p in self.y.fixPoints if not np.isinf(p) and domain.yi <= p <= domain.yf]
        self.z.fixPoints = [p for p in self.z.fixPoints if not np.isinf(p) and domain.zi <= p <= domain.zf]
#class MeshAxis:
#    def __init__(self, axisType, axisValues):
#        self.axisType = axisType  # Type of axis (e.g., 'X', 'Y', 'Z')
#        self.axisValues = axisValues  # Values along the axis
class MeshAxis:
    def __init__(self):
        self.type = 'attractors'
        self.attractor_points = np.array([])
        self.attractor_strength = np.array([])
        self.attractor_size = np.array([])
        self.attractor_regions = np.array([])
        self.match_fixed = 2
        self.periodic = False
        self.fix_periodic_domain_size = False
        self.extra_refinement = 0
        self.d0 = None
        self.buffer_i = Buffer()  # Buffer for initial part
        self.buffer_f = Buffer()  # Buffer for final part
        self.n = None



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
        self.resume = None
    
    def add_extra_region(self, location, size, X):
        self.extra_region.append({"location": location, "size": size, "X": X})


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
        self.name = 'boundaryLayerAdiabatic'
        self.initial_type = 'blasius'
        self.initial_flowFile = None
        self.initial_meshFile = None
        self.initial_blasiusFit = None
        self.initial_U0 = None
        self.initial_addNoise = None
        self.initial_meanFile = None
        self.initial_changeMach = None
        self.initial_noiseType = None
        self.initial_noiseCenter = None
        self.initial_noiseSigma = None
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


class Flow:
    def __init__(self, U, V, W, R, E):
        self.U = U
        self.V = V
        self.W = W
        self.R = R
        self.E = E
        self.t = None

######### FUNCTIONS

#Should be working fine | Used in getMatrixTypeBlocks.m, preprocessing.m, calcInstability.m
def get_domain_slices2(n, p):
    """
    This function computes the size of each slice the same way the 2decomp library does.
    It is used to correctly distribute the info across the processes.
    The distribution is as even as possible. If an uneven distribution is needed, 
    extra nodes are placed first in the last slices.
    For example, if 10 nodes are divided across 3 slices, the division would be 3, 3, 4.
    """
    if n == 1:
        # If n is 1, there's only one slice from index 0 to 0 in Python indexing
        return np.array([[0], [0]])

    # Compute base size for each slice
    nPointsBase = n // p
    nCeil = n - nPointsBase * p

    # Initialize nPoints with base size
    nPoints = np.ones(p, dtype=int) * nPointsBase

    # Distribute extra nodes to the last slices
    nPoints[-nCeil:] += 1

    # Compute cumulative sum to get the ending indices of slices
    slices_end = np.cumsum(nPoints)

    # The starting index of the first slice is 0 (Python index starts at 0)
    slices_start = np.zeros(p, dtype=int)
    slices_start[1:] = slices_end[:-1] + 1

    # Combine start and end indices into a 2D array (rows: [start, end])
    slices = np.vstack((slices_start, slices_end - 1))  # Adjusted to 0-indexing

    return slices


def get_domain_slices(n, p):
    # Esta função calcula o tamanho de cada fatia da mesma forma que a biblioteca 2decomp faz
    # É usada para distribuir corretamente as informações entre os processos
    # A distribuição é a mais uniforme possível. Se uma distribuição desigual for necessária, 
    # os nós extras são colocados primeiro nas últimas fatias
    # Por exemplo, se 10 nós são divididos em 3 fatias, a divisão seria 3 3 4
    
    if n == 1:
        return np.array([[0], [0]])
    
    nPointsBase = n // p
    nCeil = n - nPointsBase * p
    nPoints = np.ones(p, dtype=int) * nPointsBase
    nPoints[-nCeil:] = nPointsBase + 1
    
    slices = np.zeros((2, p), dtype=int)
    slices[1, :] = np.cumsum(nPoints) - 1
    slices[0, 0] = 0
    slices[0, 1:] = slices[1, :-1] + 1
    
    return slices

# Helper function for creating empty mesh and flow parameter classes in other files
def create_empty_classes():
    return Mesh(0, 0, 0), FlowParameters(0, 0, 0, 0, 0), Time(0, 0, 0, '', 0, 0), NumericalMethods('', [0, 0, 0]), Matrices(None, None, None, [], [])


def generateInitialFlow(mesh, flowParameters, initialFlow, walls, flowName):
    # Define each type of initial flow
    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    X, Y, Z = mesh.X, mesh.Y, mesh.Z

    gamma = flowParameters.gamma
    Ma = flowParameters.Ma
    Re = flowParameters.Re

    E0 = 1 / ((gamma**2 - gamma) * Ma**2)

    if initialFlow.initial_type == 'uniform':
        U = np.ones((nx, ny, nz))
        V = np.zeros((nx, ny, nz))
        W = np.zeros((nx, ny, nz))
        R = np.ones((nx, ny, nz))
        E = np.ones((nx, ny, nz)) * E0

        y0ind = np.argmin(np.abs(Y))

        if flowName is None or 'boundaryLayer' in flowName:
            U[:, :y0ind + 1, :] = 0

        if hasattr(initialFlow, 'U0'):
            U = initialFlow.U0 * U

    elif initialFlow.initial_type == 'poiseuille':
        U = np.zeros((nx, ny, nz))
        V = np.zeros((nx, ny, nz))
        W = np.zeros((nx, ny, nz))
        R = np.ones((nx, ny, nz))
        E = np.ones((nx, ny, nz)) * E0

        eta = (Y - Y[0]) / (Y[-1] - Y[0])

        u0 = flowParameters.U0
        u1 = flowParameters.lowerWallVelocity
        u2 = flowParameters.upperWallVelocity

        U = (-6*u0 + 3*u1 + 3*u2) * eta**2 + (6*u0 - 4*u1 - 2*u2) * eta + u1

    elif initialFlow.initial_type == 'blasius':
        ybl = np.arange(0, 10, 0.0001)
        ubl = blasius(ybl)  # Assumindo que a função blasius já está definida externamente
        thetabl = 0.664155332943009

        R = np.ones((nx, ny, nz))
        U = np.ones((nx, ny))
        V = np.zeros((nx, ny, nz))
        W = np.zeros((nx, ny, nz))
        E = np.ones((nx, ny, nz)) * E0

        Y0 = np.zeros(nx)
        if initialFlow.initial_blasiusFit != None:
            walls2D = np.any(walls, axis=2)
            for i in range(nx):
                Y0[i] = Y[np.argmax(walls2D[i, :] == 0)]
            for i in range(1, nx):
                dx = X[i] - X[i-1]
                if Y0[i-1] - Y0[i] > dx * initialFlow.initial_blasiusFit:
                    Y0[i] = Y0[i-1] - dx * initialFlow.initial_blasiusFit
            for i in range(nx-2, -1, -1):
                dx = X[i+1] - X[i]
                if Y0[i+1] - Y0[i] > dx * initialFlow.initial_blasiusFit:
                    Y0[i] = Y0[i+1] - dx * initialFlow.initial_blasiusFit

        for i in range(nx):
            xi = X[i]
            if xi > 0:
                theta = 0.664 * np.sqrt(xi / Re)
                U_interp = interp1d(ybl*theta/thetabl + Y0[i], ubl, kind='linear', fill_value='extrapolate')
                U[i, :] = U_interp(Y)
                U[i, Y < Y0[i]] = 0

        U = np.clip(U, 0, 1)
        U = np.tile(U[:, :, np.newaxis], (1, 1, nz))

    elif initialFlow.initial_type == 'compressibleBL_isothermal':
        compressibleBL_flow = calcCompressibleBL(flowParameters, False, mesh)
        U, V, W, E, R = compressibleBL_flow.U, compressibleBL_flow.V, compressibleBL_flow.W, compressibleBL_flow.E, compressibleBL_flow.R

    elif initialFlow.initial_type == 'compressibleBL_adiabatic':
        compressibleBL_flow = calcCompressibleBL(flowParameters, True, mesh)
        U, V, W, E, R = compressibleBL_flow.U, compressibleBL_flow.V, compressibleBL_flow.W, compressibleBL_flow.E, compressibleBL_flow.R

    elif initialFlow.initial_type == 'file':
        if initialFlow.flowFile.endswith('/'):
            nStep = checkPreviousRun(initialFlow.flowFile[:-1])  # Assumindo que checkPreviousRun já está definido
            if nStep is not None:
                initialFlow.initial_flowFile = f"{initialFlow.initial_flowFile}flow_{nStep:010d}.npy"
            else:
                initialFlow.initial_flowFile = f"{initialFlow.initial_flowFile}baseflow.npy"

        flowFile = np.load(initialFlow.initial_flowFile, allow_pickle=True).item()  # Assumindo que flowFile contém um dicionário

        if initialFlow.initial_meshFile != None:
            if initialFlow.initial_meshFile.endswith('/'):
                initialFlow.initial_meshFile = f"{initialFlow.initial_meshFile}mesh.npy"

            meshFile = np.load(initialFlow.initial_meshFile, allow_pickle=True).item()
            Xfile, Yfile, Zfile = meshFile['X'], meshFile['Y'], meshFile['Z']

            Ufile, Vfile, Wfile, Rfile, Efile = flowFile['U'], flowFile['V'], flowFile['W'], flowFile['R'], flowFile['E']

            Ufile[np.isnan(Ufile)] = 0
            Vfile[np.isnan(Vfile)] = 0
            Wfile[np.isnan(Wfile)] = 0
            Rfile[np.isnan(Rfile)] = 1
            Efile[np.isnan(Efile)] = E0

            Xmesh, Ymesh, Zmesh = np.meshgrid(X, Y, Z, indexing='ij')

            Xmesh = np.clip(Xmesh, Xfile[0], Xfile[-1])
            Ymesh = np.clip(Ymesh, Yfile[0], Yfile[-1])
            Zmesh = np.clip(Zmesh, Zfile[0], Zfile[-1])

            if nz == 1 or len(Zfile) == 1:
                U = RegularGridInterpolator((Xfile, Yfile), Ufile)(np.array([Xmesh, Ymesh]).T)
                V = RegularGridInterpolator((Xfile, Yfile), Vfile)(np.array([Xmesh, Ymesh]).T)
                W = RegularGridInterpolator((Xfile, Yfile), Wfile)(np.array([Xmesh, Ymesh]).T)
                R = RegularGridInterpolator((Xfile, Yfile), Rfile)(np.array([Xmesh, Ymesh]).T)
                E = RegularGridInterpolator((Xfile, Yfile), Efile)(np.array([Xmesh, Ymesh]).T)
            else:
                U = RegularGridInterpolator((Xfile, Yfile, Zfile), Ufile)(np.array([Xmesh, Ymesh, Zmesh]).T)
                V = RegularGridInterpolator((Xfile, Yfile, Zfile), Vfile)(np.array([Xmesh, Ymesh, Zmesh]).T)
                W = RegularGridInterpolator((Xfile, Yfile, Zfile), Wfile)(np.array([Xmesh, Ymesh, Zmesh]).T)
                R = RegularGridInterpolator((Xfile, Yfile, Zfile), Rfile)(np.array([Xmesh, Ymesh, Zmesh]).T)
                E = RegularGridInterpolator((Xfile, Yfile, Zfile), Efile)(np.array([Xmesh, Ymesh, Zmesh]).T)

            if initialFlow.initial_changeMach != None:
                if meshFile['flowParameters']['Ma'] != Ma:
                    print(f"Initial flow file has a Mach number of {meshFile['flowParameters']['Ma']} which will be changed to {Ma}")
                    E *= meshFile['flowParameters']['Ma']**2 / Ma**2
        else:
            U, V, W, R, E = flowFile['U'], flowFile['V'], flowFile['W'], flowFile['R'], flowFile['E']

            if (nx, ny, nz) != U.shape:
                raise ValueError(f"Mesh size in initial flow file is not consistent with current mesh {U.shape} -> {(nx, ny, nz)}")

            if nz == 1 and U.shape[2] == 1:
                U, V, W, R, E = [np.tile(arr, (1, 1, nz)) for arr in [U, V, W, R, E]]

    if initialFlow.initial_addNoise != None:
        if initialFlow.initial_noiseType == None or initialFlow.initial_noiseType == 'rand':
            noiseU = initialFlow.initial_addNoise * np.random.randn(nx, ny, nz)
            noiseV = initialFlow.initial_addNoise * np.random.randn(nx, ny, nz)
            noiseW = initialFlow.initial_addNoise * np.random.randn(nx, ny, nz)
            noiseR = initialFlow.initial_addNoise * np.random.randn(nx, ny, nz)
            noiseE = initialFlow.initial_addNoise * np.random.randn(nx, ny, nz)
        elif initialFlow.initial_noiseType == 'uniform':
            noiseU = initialFlow.initial_addNoise * np.ones((nx, ny, nz))
            noiseV = initialFlow.initial_addNoise * np.ones((nx, ny, nz))
            noiseW = initialFlow.initial_addNoise * np.ones((nx, ny, nz))
            noiseR = initialFlow.initial_addNoise * np.ones((nx, ny, nz))
            noiseE = initialFlow.initial_addNoise * np.ones((nx, ny, nz))

        if initialFlow.initial_noiseCenter != None:
            if initialFlow.initial_noiseSigma == None:
                initialFlow.initial_noiseSigma = [(X[-1] - X[0])**2 / 10, (Y[-1] - Y[0])**2 / 10, (Z[-1] - Z[0])**2 / 10]
            
            x0 = initialFlow.initial_noiseCenter[0]
            y0 = initialFlow.initial_noiseCenter[1]
            z0 = initialFlow.initial_noiseCenter[2]

            sigmaX = initialFlow.initial_noiseSigma[0]
            sigmaY = initialFlow.initial_noiseSigma[1]
            sigmaZ = initialFlow.initial_noiseSigma[2]
        else:
            x0, y0, z0 = 0, 0, 0
            sigmaX, sigmaY, sigmaZ = np.inf, np.inf, np.inf

        if nz == 1:
            sigmaZ = np.inf

        radius = (X.T - x0)**2 / sigmaX + (Y - y0)**2 / sigmaY
        radius = np.add(radius, (np.transpose(Z, (0, 2, 1)) - z0)**2 / sigmaZ)
        noiseGaussian = np.exp(-radius)

        U += noiseU * noiseGaussian
        V += noiseV * noiseGaussian
        if nz > 1:
            W += noiseW * noiseGaussian
        R += noiseR * noiseGaussian
        E += noiseE * noiseGaussian

    return Flow(U, V, W, R, E)

def blasius(y):
    # Condições iniciais
    f0 = [0, 0, 0.33204312]

    # Resolver a EDO usando odeint
    sol = odeint(blasius_eq, f0, y)

    # Extrair a segunda coluna da solução
    u = sol[:, 1]
    
    return u

def blasius_eq(f, t):
    # Definir o sistema de equações
    dfdt = np.zeros(3)
    dfdt[2] = -0.5 * f[0] * f[2]
    dfdt[1] = f[2]
    dfdt[0] = f[1]
    
    return dfdt

def calcCompressibleBL(flowParameters, adiabWall, mesh):
    # Parâmetros de fluxo
    Re = flowParameters.Re
    Minf = flowParameters.Ma
    Pr = flowParameters.Pr
    Tinf = flowParameters.T0
    gamma = flowParameters.gamma

    # Variáveis auxiliares
    xR = Re / 1.7208**2  # Referência de x
    E0 = 1 / (gamma * (gamma - 1) * Minf**2)
    
    # Temperatura da parede
    Twall = 1
    if hasattr(flowParameters, 'Twall'):
        Twall = flowParameters.Twall / Tinf

    # Resolver camada limite compressível
    sol = solve_compressibleBL(Minf, Pr, Tinf, Twall, gamma, adiabWall)

    eta_xR = sol.eta
    U_xR = sol.f_p
    R_xR = 1.0 / sol.rbar
    E_xR = sol.rbar * E0
    V_xR = -(sol.rbar) * (sol.f - eta_xR * sol.f_p) * (1.7208 / np.sqrt(2)) * (1 / Re)

    # Calcular espessura da camada limite
    y_xR = eta_xR * np.sqrt(2) / 1.7208
    dS_xR = np.trapz(1 - U_xR * R_xR / (U_xR[-1] * R_xR[-1]), y_xR)

    print("Initial Flow: Compressible BL")
    if adiabWall:
        print(f"Adiabatic Wall, (dT/dy)_wall = 0, (Twall/TInf) = {E_xR[0] / E0:.3f}")
    else:
        print(f"Isothermal Wall, T_wall = constant, (Twall/TInf) = {E_xR[0] / E0:.3f}")
    
    print(f"BL thickness, (effective) / (incomp. BL, Blasius) = {dS_xR:.4f}")

    # Parâmetros da malha
    X = mesh.X
    Y = mesh.Y
    Z = mesh.Z
    nx, ny, nz = len(X), len(Y), len(Z)

    # Inicializar arrays de fluxo
    R = np.ones((nx, ny))
    U = np.ones((nx, ny))
    V = np.zeros((nx, ny))
    W = np.zeros((nx, ny))
    E = np.ones((nx, ny)) * E0

    # Encontrar índices
    indX = np.where(X > 0)[0]
    indY = np.where(Y >= 0)[0]

    for ix in indX:
        y_xL = y_xR * np.sqrt(X[ix] / xR)

        interp_U = interp1d(y_xL, U_xR, kind='linear', fill_value=U_xR[-1], bounds_error=False)
        interp_V = interp1d(y_xL, V_xR, kind='linear', fill_value=V_xR[-1], bounds_error=False)
        interp_R = interp1d(y_xL, R_xR, kind='linear', fill_value=R_xR[-1], bounds_error=False)
        interp_E = interp1d(y_xL, E_xR, kind='linear', fill_value=E_xR[-1], bounds_error=False)

        U[ix, indY] = interp_U(Y[indY])
        V[ix, indY] = interp_V(Y[indY])
        R[ix, indY] = interp_R(Y[indY])
        E[ix, indY] = interp_E(Y[indY])

    # Aplicar condições de contorno para Y < 0
    indY_neg = np.where(Y < 0)[0]
    U[:, indY_neg] = np.tile(U[:, Y == 0], (1, len(indY_neg)))
    V[:, indY_neg] = np.tile(V[:, Y == 0], (1, len(indY_neg)))
    R[:, indY_neg] = np.tile(R[:, Y == 0], (1, len(indY_neg)))
    E[:, indY_neg] = np.tile(E[:, Y == 0], (1, len(indY_neg)))

    # Replicar para terceira dimensão Z
    U = np.tile(U[:, :, np.newaxis], (1, 1, nz))
    V = np.tile(V[:, :, np.newaxis], (1, 1, nz))
    W = np.tile(W[:, :, np.newaxis], (1, 1, nz))
    R = np.tile(R[:, :, np.newaxis], (1, 1, nz))
    E = np.tile(E[:, :, np.newaxis], (1, 1, nz))
    
    class Flow:
        def __init__(self,U, V, W, R, E):
            self.U = U
            self.V = V
            self.W = W
            self.R = R
            self.E = E
    # Retornar fluxo
    return Flow(U, V, W, R, E)

# Função para resolver a camada limite compressível
def solve_compressibleBL(Minf, Pr, Tinf, Twall, Gamma, adiabWall):
    C2 = 110  # Sutherland Coefficient [Kelvin]
    lim = 10  # Simula lim -> inf
    N = 500  # Número de pontos
    h = lim / N  # Delta y
    delta = 1e-10  # Número pequeno para o método de tiro
    eps = 1e-9

    adi = 1 if adiabWall else 0

    # Inicializando
    y1 = np.zeros(N + 1)  # f
    y2 = np.zeros(N + 1)  # f'
    y3 = np.zeros(N + 1)  # f''
    y4 = np.zeros(N + 1)  # rho(eta)
    y5 = np.zeros(N + 1)  # rho(eta)'
    eta = custom_linspace(0, lim, N + 1)  # Iteração de eta até o infinito

    # Condições de contorno e chute inicial
    if adi == 1:
        y1[0] = 0
        y2[0] = 0
        y5[0] = 0
        alfa0 = 0.1  # Chute inicial
        beta0 = 3  # Chute inicial
    else:
        y1[0] = 0
        y2[0] = 0
        y4[0] = Twall
        alfa0 = 0.1  # Chute inicial
        beta0 = 3  # Chute inicial

    # Método de tiro
    for _ in range(100000):
        if adi == 1:
            y1[0] = 0
            y2[0] = 0
            y5[0] = 0
            y3[0] = alfa0
            y4[0] = beta0
        else:
            y1[0] = 0
            y2[0] = 0
            y4[0] = Twall
            y3[0] = alfa0
            y5[0] = beta0

        y1, y2, y3, y4, y5 = RK(eta, h, y1, y2, y3, y4, y5, C2, Tinf, Minf, Pr, Gamma)

        y2old = y2[-1]
        y4old = y4[-1]

        if adi == 1:
            y1[0] = 0
            y2[0] = 0
            y5[0] = 0
            y3[0] = alfa0 + delta
            y4[0] = beta0
        else:
            y1[0] = 0
            y2[0] = 0
            y4[0] = Twall
            y3[0] = alfa0 + delta
            y5[0] = beta0

        y1, y2, y3, y4, y5 = RK(eta, h, y1, y2, y3, y4, y5, C2, Tinf, Minf, Pr, Gamma)

        y2new1 = y2[-1]
        y4new1 = y4[-1]

        if adi == 1:
            y1[0] = 0
            y2[0] = 0
            y5[0] = 0
            y3[0] = alfa0
            y4[0] = beta0 + delta
        else:
            y1[0] = 0
            y2[0] = 0
            y4[0] = Twall
            y3[0] = alfa0
            y5[0] = beta0 + delta

        y1, y2, y3, y4, y5 = RK(eta, h, y1, y2, y3, y4, y5, C2, Tinf, Minf, Pr, Gamma)

        y2new2 = y2[-1]
        y4new2 = y4[-1]

        a11 = (y2new1 - y2old) / delta
        a21 = (y4new1 - y4old) / delta
        a12 = (y2new2 - y2old) / delta
        a22 = (y4new2 - y4old) / delta
        r1 = 1 - y2old
        r2 = 1 - y4old

        dalfa = (a22 * r1 - a12 * r2) / (a11 * a22 - a12 * a21)
        dbeta = (a11 * r2 - a21 * r1) / (a11 * a22 - a12 * a21)

        alfa0 += dalfa
        beta0 += dbeta

        if abs(y2[-1] - 1) < eps and abs(y4[-1] - 1) < eps:
            Truey2 = y2[0]
            Truey4 = y4[0]
            break

    # Cálculo do eixo x usando simpson para interpolação
    xaxis = np.zeros_like(eta)
    for i in range(1, len(eta)):
        xaxis[i] = simpson(y4[:i+1], eta[:i+1])

    # Criando a solução com uma classe
    class Solution:
        def __init__(self, eta, f, f_p, f_pp, rbar, rbar_p):
            self.eta = eta
            self.f = f
            self.f_p = f_p
            self.f_pp = f_pp
            self.rbar = rbar
            self.rbar_p = rbar_p

    sol = Solution(eta=xaxis, f=y1, f_p=y2, f_pp=y3, rbar=y4, rbar_p=y5)
    
    return sol

# Método de Runge-Kutta de quarta ordem
def RK(eta, h, y1, y2, y3, y4, y5, C2, Tinf, Minf, Pr, gamma):
    for i in range(len(eta) - 1):
        # Cálculo das variáveis de Runge-Kutta
        k11 = y2[i]
        k21 = y3[i]
        k31 = Y3(y1[i], y3[i], y4[i], y5[i], C2, Tinf)
        k41 = y5[i]
        k51 = Y5(y1[i], y3[i], y4[i], y5[i], C2, Tinf, Minf, Pr, gamma)

        # Atualizações intermediárias
        k12 = y2[i] + 0.5 * h * k21
        k22 = y3[i] + 0.5 * h * k31
        k32 = Y3(y1[i] + 0.5 * h * k11, y3[i] + 0.5 * h * k31, y4[i] + 0.5 * h * k41, y5[i] + 0.5 * h * k51, C2, Tinf)
        k42 = y5[i] + 0.5 * h * k51
        k52 = Y5(y1[i] + 0.5 * h * k11, y3[i] + 0.5 * h * k31, y4[i] + 0.5 * h * k41, y5[i] + 0.5 * h * k51, C2, Tinf, Minf, Pr, gamma)

        # Atualizações finais
        y5[i+1] = y5[i] + (1/6) * (k51 + 2 * k52 + 2 * k52 + k51) * h
        y4[i+1] = y4[i] + (1/6) * (k41 + 2 * k42 + 2 * k42 + k41) * h
        y3[i+1] = y3[i] + (1/6) * (k31 + 2 * k32 + 2 * k32 + k31) * h
        y2[i+1] = y2[i] + (1/6) * (k21 + 2 * k22 + 2 * k22 + k21) * h
        y1[i+1] = y1[i] + (1/6) * (k11 + 2 * k12 + 2 * k12 + k11) * h

    return y1, y2, y3, y4, y5

def Y1(y2):
    return y2

# Função Y2
def Y2(y3):
    return y3

# Função Y3
def Y3(y1, y3, y4, y5, C2, Tinf):
    RHS = -y3 * ((y5 / (2 * y4)) - (y5 / (y4 + C2 / Tinf))) \
          - y1 * y3 * ((y4 + C2 / Tinf) / (y4**0.5 * (1 + C2 / Tinf)))
    return RHS

# Função Y4
def Y4(y5):
    return y5

# Função Y5
def Y5(y1, y3, y4, y5, C2, Tinf, Minf, Pr, Gamma):
    RHS = -y5**2 * ((0.5 / y4) - (1 / (y4 + C2 / Tinf))) \
          - Pr * y1 * y5 / y4**0.5 * (y4 + C2 / Tinf) / (1 + C2 / Tinf) \
          - (Gamma - 1) * Pr * Minf**2 * y3**2
    return RHS

def checkPreviousRun(caseName):
        """
        Esta função verifica se há arquivos de execução anteriores.
        caseName é a pasta a ser verificada.
        nStep é o último passo de tempo encontrado.
        nx, ny, e nz são os tamanhos da malha no arquivo salvo.
        Se nenhum arquivo for encontrado, arrays vazios são retornados.

        #TODO:A estrutura condicional nargout > 1 foi adaptada para Python com base em 
        como o código lida com múltiplas saídas, e pode ser ajustada dependendo 
        do contexto do código principal.

        """
        # Lista todos os arquivos no diretório
        allFiles = os.listdir(caseName)

        # Lista para armazenar os arquivos válidos
        caseFiles = []

        # Procura por arquivos que correspondem ao padrão 'flow_*.npy'
        for name in allFiles:
            if len(name) == 19 and re.search(r'flow_\d*.npy', name):
                caseFiles.append(name)

        # Se nenhum arquivo for encontrado, retorna valores vazios
        if not caseFiles:
            return None, None, None, None

        # Extrai o número de passos de tempo dos arquivos encontrados
        nSteps = [int(re.search(r'\d+', name).group()) for name in caseFiles]

        # Encontra o maior passo de tempo
        nStep = max(nSteps)

        #TODO: Aqui precisa ser aprimorado
        # Se o número de saídas for maior que 1, carrega o arquivo e obtém as dimensões
        #if hasattr(self, 'nargout') and self.nargout > 1:
        #    file_path = os.path.join(caseName, f'flow_{nStep:010d}.npy')
        #    fileObject = np.load(file_path, allow_pickle=True).item()
        #    nx, ny, nz = fileObject['U'].shape
        #    return nStep, nx, ny, nz
        
        file_path = os.path.join(caseName, f'flow_{nStep:010d}.npy')
        fileObject = np.load(file_path, allow_pickle=True).item()
        nx, ny, nz = fileObject.U.shape

        # Se não for necessário retornar as dimensões, apenas retorna o nStep
        return nStep, nx, ny, nz

def custom_linspace(x, y, num):
    if num == 1:
        return np.array([y])  # Retorna y em vez de x quando num = 1
    else:
        return np.linspace(x, y, num)  # Comportamento padrão para num > 1
    
def custom_range(N):
    if N > 1:
        for k in range(N):
            yield k