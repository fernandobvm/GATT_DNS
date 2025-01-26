import os
import numpy as np
from source.library import *
from Simulation import runDNS

## Case name
case_name = 'baseflow_gap_Re950_M01_h10_LR18'

## Domain decomposition
p_row 	 = 4
p_col    = 1

## Flow parameters
Re = 950
Ma = 0.1
Pr = 0.71
gamma = 1.4
T0 = 300

flow_params = FlowParameters(Re, Ma, Pr, gamma, T0)

# Domain parameters
xi = -50
xf = 500
yi = -0.10
yf = 16
zi =  0
zf =  1

domain = Domain(xi, xf, yi, yf, zi, zf)

# Cavity parameters
x1 = 311.73
x2 = 329.91
cav1 = Cavity([x1, x2], [yi, 0], [-np.inf, np.inf])

# Disturbance parameters
disturb = Disturbance(x_range=[165.00, 182.50], y_range=[0, 0], z_range=[-np.inf, np.inf], var='U', disturb_type='packet_2d', par=[0.00475, 40, 0.004167696485], active=False)

# Flow type
flow_type = FlowType()
flow_type.name = "boundaryLayerIsothermal"
flow_type.initial_type = "blasius"
flow_type.cav.append(cav1)
flow_type.disturb.append(disturb)

# Mesh parameters
mesh = Mesh()

mesh.x.n = 925
mesh.y.n = 207
mesh.z.n = 1

mesh.x.type = 'file'
mesh.x.file = './meshes/meshX_Re950_npts925_dxMin01.mat'

mesh.y.type = 'file'
mesh.y.file = './meshes/meshY_npts207_dyMin005_gap_h10.mat'

mesh.z.type = 'uniform'

mesh.x.fileCalcBuffer 	   = True
mesh.x.buffer_i.n 	       = 0
mesh.x.buffer_f.n 	       = 75
mesh.x.buffer_i.transition = 1  # Não estar explícitamente definido aqui assume um comportamento diferente da versão matlab.
mesh.x.buffer_f.transition = 1  # Não estar explícitamente definido aqui assume um comportamento diferente da versão matlab.
mesh.x.buffer_f.type 	   = 'sigmoid'
mesh.x.buffer_f.stretching = 8.6

mesh.y.fileCalcBuffer	   = True
mesh.y.buffer_i.n 	       = 0
mesh.y.buffer_f.n          = 20
mesh.y.buffer_i.transition = 1  # Não estar explícitamente definido aqui assume um comportamento diferente da versão matlab.
mesh.y.buffer_f.transition = 1  # Não estar explícitamente definido aqui assume um comportamento diferente da versão matlab.
mesh.y.buffer_f.type 	   = 'sigmoid'
mesh.y.buffer_f.stretching = 3.65

mesh.z.buffer_i.n 		= 0 
mesh.z.buffer_f.n 		= 0

mesh.x.match_fixed = False
mesh.y.match_fixed = True
mesh.z.match_fixed = True

mesh.x.periodic = False
mesh.y.periodic = False
mesh.z.periodic = True

mesh.x.fix_periodic_domain_size = False
mesh.y.fix_periodic_domain_size = False
mesh.z.fix_periodic_domain_size = True

mesh.x.extra_refinement = 0
mesh.y.extra_refinement	= 0
mesh.z.extra_refinement = 0

# Time control
time = Time(control='dt', dt=0.7*6.178885716288634e-4, max_cfl=1.3, qtimes=6690, tmax=6690*1024*2, nStep= 0)

logAll = 25

#SFD
sfd = SFD(type = 2, X = 0.05, Delta = 10, applyY= True)
# Numerical methods
num_methods = NumericalMethods(spatial_derivs='SL6O3', spatial_derivs_buffer='EX2', time_stepping='RK4', neumann_order=6, neumann2_order=2, spatial_filter_strength=0.49, spatial_filter_time=0, filter_directions= [1 , 1, 1], filter_borders=False, sfd=sfd)

caseFile = os.path.basename(__file__)
# runDNS(case_name, flow_params, domain, flow_type, mesh, time, num_methods, p_row, p_col, caseFile, logAll)