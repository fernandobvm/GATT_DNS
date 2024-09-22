import numpy as np

def calcSDFregion(mesh, num_methods):
    SFD_X = np.ones((mesh.nx, mesh.ny, mesh.nz))

    # Aplicar valores padrão
    num_methods.SFD.applyX = getattr(num_methods.SFD, 'applyX', 2)
    num_methods.SFD.applyY = getattr(num_methods.SFD, 'applyY', 2)
    num_methods.SFD.applyZ = getattr(num_methods.SFD, 'applyZ', 2)

    # Aplicar modificações em X
    if num_methods.SFD.applyX == 2 or num_methods.SFD.applyX == -1:
        for i in range(mesh.x.buffer_i.n):
            SFD_X[i, :, :] *= (0.5 - 0.5 * np.cos(np.pi * (i) / (mesh.x.buffer_i.n)))
    
    if num_methods.SFD.applyX == 2 or num_methods.SFD.applyX == 1:
        for i in range(mesh.x.buffer_f.n):
            SFD_X[mesh.nx - i - 1, :, :] *= (0.5 - 0.5 * np.cos(np.pi * (i) / (mesh.x.buffer_f.n)))

    # Aplicar modificações em Y
    if num_methods.SFD.applyY == 2 or num_methods.SFD.applyY == -1:
        for j in range(mesh.y.buffer_i.n):
            SFD_X[:, j, :] *= (0.5 - 0.5 * np.cos(np.pi * (j) / (mesh.y.buffer_i.n)))
    
    if num_methods.SFD.applyY == 2 or num_methods.SFD.applyY == 1:
        for j in range(mesh.y.buffer_f.n):
            SFD_X[:, mesh.ny - j - 1, :] *= (0.5 - 0.5 * np.cos(np.pi * (j) / (mesh.y.buffer_f.n)))

    # Aplicar modificações em Z
    if num_methods.SFD.applyZ == 2 or num_methods.SFD.applyZ == -1:
        for k in range(mesh.z.buffer_i.n):
            SFD_X[:, :, k] *= (0.5 - 0.5 * np.cos(np.pi * (k) / (mesh.z.buffer_i.n)))

    if num_methods.SFD.applyZ == 2 or num_methods.SFD.applyZ == 1:
        for k in range(mesh.z.buffer_f.n):
            SFD_X[:, :, mesh.nz - k - 1] *= (0.5 - 0.5 * np.cos(np.pi * (k) / (mesh.z.buffer_f.n)))

    SFD_X = num_methods.SFD.X * (1 - SFD_X)

    # Aplicar regiões extras
    if num_methods.SFD.extraRegion:
        for ER in num_methods.SFD.extraRegion:
            # Obter raio do ponto central
            R = np.sqrt(((mesh.X - ER.location[0])**2) / (ER.size[0]**2) +
                         ((mesh.Y - ER.location[1])**2) / (ER.size[1]**2) +
                         ((mesh.Z.transpose(0, 2, 1) - ER.location[2])**2) / (ER.size[2]**2))
            R[R > 1] = 1
            R = 0.5 + 0.5 * np.cos(np.pi * R)
            SFD_X += ER.X * R

    return SFD_X