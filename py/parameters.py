import numpy as np
from source.library import *

# Main code
delta = 1
Re = 600
Ma = 0.5
L = 10
D = 5
x1 = delta**2 * Re / (1.72**2 * delta**2)
x2 = x1 + L
x_end = x2 + 200
delta99_end = 5 * x_end / np.sqrt(Re * x_end)
y_end = 5 * delta99_end

# Case name
case_name = f'Re{Re}-Ma{str(Ma).replace(".", "")}-L{L}-D{D}'

# Flow parameters
flow_params = FlowParameters(Re=Re, Ma=Ma)

# Domain parameters
domain = Domain(xi=0, xf=x_end, yi=-D, yf=y_end, zi=0, zf=1)

# Flow type - #TODO: blasiusFit,flowFile e meshFile
flow_type = FlowType()
flow_type.cav.append(Cavity([x1, x2], [-D, 0], [-np.inf, np.inf]))

disturb = Disturbance(x_range=[25, 50], y_range=[0, 0], z_range=[-np.inf, np.inf], var='V', disturb_type='packet_2d', par=[0.02, 50, 1e-5])
flow_type.disturb.append(disturb)

# Mesh parameters
mesh = Mesh()
mesh.x.d0 = 4
mesh.y.d0 = 1
mesh.z.n = 1

mesh.x.type = 'attractors'
x1 = flow_type.cav[0].x[0]
x2 = flow_type.cav[0].x[1]
L = x2 - x1
mesh.x.attractor_points = np.array([])
mesh.x.attractor_strength = np.array([])
mesh.x.attractor_size = np.array([])
mesh.x.attractor_regions = np.array([
    [x1-100, x1-50, x2+50, x2+100, 3],
    [x1-20, x1, x2+10, x2+20, 16],
    [x2-5, x2-0.1, x2, x2+2, 30],
    [x1-2, x1, x1+0.1, x1+5, 20],
    [-30, -10, 10, 30, 1]
])
mesh.x.match_fixed = 2
mesh.x.periodic = False
mesh.x.fix_periodic_domain_size = False
mesh.x.extra_refinement = 0

mesh.y.type = 'attractors'
mesh.y.attractor_points = np.array([])
mesh.y.attractor_strength = np.array([])
mesh.y.attractor_size = np.array([])
mesh.y.attractor_regions = [-2*D, -D/10, D/10, 2*delta99_end, 12]
mesh.y.match_fixed = 2
mesh.y.periodic = False
mesh.y.fix_periodic_domain_size = False
mesh.y.extra_refinement = 0

mesh.z.type = 'uniform'
#mesh.z.file = 'baseflows/3Dtest/z.dat'
mesh.z.match_fixed = 2
mesh.z.periodic = True
mesh.z.fix_periodic_domain_size = True
mesh.z.extra_refinement = 0

mesh.x.buffer_i.n = 40
mesh.x.buffer_i.type = 'sigmoid'
mesh.x.buffer_i.stretching = 30
mesh.x.buffer_i.transition = 0.2
#mesh.x.buffer_i.ramp = 20

mesh.x.buffer_f.n = 30
mesh.x.buffer_f.type = 'sigmoid'
mesh.x.buffer_f.stretching = 15
mesh.x.buffer_f.transition = 0.2
#mesh.x.buffer_f.ramp = 20

mesh.y.buffer_i.n = 0

mesh.y.buffer_f.n = 40
mesh.y.buffer_f.type = 'sigmoid'
mesh.y.buffer_f.stretching = 30
mesh.y.buffer_f.transition = 0.2
#mesh.y.buffer_f.ramp = 20
#mesh.y.buffer_f.type = 'exponential'
#mesh.y.buffer_f.stretching = 0.1

mesh.z.buffer_i.n = 0
mesh.z.buffer_f.n = 0


# Time control
time = Time(control='cfl', dt=1, max_cfl=1.3, qtimes=100, tmax=10000)

# Numerical methods
num_methods = NumericalMethods(spatial_derivs='SL6', spatial_derivs_buffer='EX4', time_stepping='RK4', neumann_order=6, neumann2_order=2, spatial_filter_strength=0.49)

# Tracked points
tracked_x = np.arange(0, domain.xf, 50)
n_probes = len(tracked_x)
mesh.tracked_points = np.vstack([
    [37.5, 0, 0], 
    [(x1 + 3 * x2) / 4, 0, 0], 
    np.column_stack([tracked_x, np.ones(n_probes), np.zeros(n_probes)])
])