class Mesh:
    def __init__(self):
        self.x_fix_points = [0]
        self.y_fix_points = [0]
        self.z_fix_points = []

    def add_fix_points(self, x, y, z):
        self.x_fix_points.extend(x)
        self.y_fix_points.extend(y)
        self.z_fix_points.extend(z)

    def unique_fix_points(self, domain):
        self.x_fix_points = sorted(set(self.x_fix_points))
        self.y_fix_points = sorted(set(self.y_fix_points))
        self.z_fix_points = sorted(set(self.z_fix_points))

        # Remove invalid points outside the domain or infinite values
        self.x_fix_points = [x for x in self.x_fix_points if domain.xi <= x <= domain.xf and not float('inf') == x]
        self.y_fix_points = [y for y in self.y_fix_points if domain.yi <= y <= domain.yf and not float('inf') == y]
        self.z_fix_points = [z for z in self.z_fix_points if domain.zi <= z <= domain.zf and not float('inf') == z]


class Domain:
    def __init__(self, xi, xf, yi, yf, zi, zf):
        self.xi = xi
        self.xf = xf
        self.yi = yi
        self.yf = yf
        self.zi = zi
        self.zf = zf


class FlowType:
    def __init__(self):
        self.cav = []
        self.rug = []
        self.disturb = []

    def add_cav(self, x, y, z):
        self.cav.append({'x': x, 'y': y, 'z': z})

    def add_rug(self, x, y, z):
        self.rug.append({'x': x, 'y': y, 'z': z})

    def add_disturb(self, x, y, z, fit_points=False, fit_points_x=False, fit_points_y=False, fit_points_z=False):
        self.disturb.append({
            'x': x, 'y': y, 'z': z,
            'fitPoints': fit_points, 'fitPointsX': fit_points_x,
            'fitPointsY': fit_points_y, 'fitPointsZ': fit_points_z
        })


# Create the mesh and domain
mesh = Mesh()
domain = Domain(xi=0, xf=10, yi=0, yf=10, zi=0, zf=10)
flow_type = FlowType()

# Simulating the logic to add points (same structure as MATLAB code)
# Add points from 'cav'
for cav in flow_type.cav:
    mesh.add_fix_points([cav['x']], [cav['y']], [cav['z']])

# Add points from 'rug'
for rug in flow_type.rug:
    mesh.add_fix_points([rug['x']], [rug['y']], [rug['z']])

# Add points from 'disturb'
for disturb in flow_type.disturb:
    if disturb['fitPoints']:
        mesh.add_fix_points([disturb['x']], [disturb['y']], [disturb['z']])
    else:
        if disturb['fitPointsX']:
            mesh.add_fix_points([disturb['x']], [], [])
        if disturb['fitPointsY']:
            mesh.add_fix_points([], [disturb['y']], [])
        if disturb['fitPointsZ']:
            mesh.add_fix_points([], [], [disturb['z']])

# If tracked points exist and should be fitted
if hasattr(mesh, 'trackedPoints') and hasattr(mesh, 'fitTrackedPoints') and mesh.fit_tracked_points:
    for point in mesh.trackedPoints:
        mesh.add_fix_points([point[0]], [point[1]], [point[2]])

# Remove duplicates and filter points within the domain
mesh.unique_fix_points(domain)
