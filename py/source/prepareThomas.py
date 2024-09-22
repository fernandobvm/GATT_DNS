import numpy as np

class Matrix:
    def __init__(self, LHS, fLHS, RHS, fRHS, types):
        self.LHS = LHS
        self.fLHS = fLHS
        self.RHS = RHS
        self.fRHS = fRHS
        self.types = types
        self.periodic = None
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.Af = None
        self.Bf = None
        self.Cf = None
        self.Df = None
        self.R = None
        self.nRHS = None
        self.Rf = None
        self.nRHSf = None
        self.nTypes = None

def prepare_thomas(matrix):
    is_periodic = matrix.LHS[0][-1, 0] != 0
    n_types = np.max(matrix.types)
    N = matrix.LHS[0].shape[0]

    # Prepare LHS
    A = np.zeros((N-1, n_types))
    B = np.zeros((N, n_types))
    C = np.zeros((N-1, n_types))
    D = np.zeros((N, n_types))

    A1 = np.zeros(n_types)
    Cn = np.zeros(n_types)

    Af = np.zeros((N-1, n_types))
    Bf = np.zeros((N, n_types))
    Cf = np.zeros((N-1, n_types))
    Df = np.zeros((N, n_types))

    A1f = np.zeros(n_types)
    Cnf = np.zeros(n_types)

    for i in range(n_types):
        A[:, i] = np.diag(matrix.LHS[i], -1)
        B[:, i] = np.diag(matrix.LHS[i])
        C[:, i] = np.diag(matrix.LHS[i], 1)

        Af[:, i] = np.diag(matrix.fLHS[i], -1)
        Bf[:, i] = np.diag(matrix.fLHS[i])
        Cf[:, i] = np.diag(matrix.fLHS[i], 1)

    if is_periodic:
        for i in range(n_types):
            A1[i] = matrix.LHS[i][0, -1]
            Cn[i] = matrix.LHS[i][-1, 0]

            A1f[i] = matrix.fLHS[i][0, -1]
            Cnf[i] = matrix.fLHS[i][-1, 0]

        A = np.delete(A, 0, axis=0)
        B = np.delete(B, 0, axis=0)
        C = np.delete(C, 0, axis=0)
        Af = np.delete(Af, 0, axis=0)
        Bf = np.delete(Bf, 0, axis=0)
        Cf = np.delete(Cf, 0, axis=0)

        for i in range(N-2):
            C[i, :] /= B[i, :]
            B[i+1, :] -= A[i, :] * C[i, :]

            Cf[i, :] /= Bf[i, :]
            Bf[i+1, :] -= Af[i, :] * Cf[i, :]

        B = 1.0 / B
        Bf = 1.0 / Bf

        A = np.vstack([A1, A])
        Af = np.vstack([A1f, Af])
        C = np.vstack([Cn, C])
        Cf = np.vstack([Cnf, Cf])

        for i in range(n_types):
            D[:, i] = np.linalg.inv(matrix.LHS[i])[0, :]
            Df[:, i] = np.linalg.inv(matrix.fLHS[i])[0, :]

    else:
        A1.fill(0)
        Cn.fill(0)
        A1f.fill(0)
        Cnf.fill(0)

        for i in range(N-1):
            C[i, :] /= B[i, :]
            B[i+1, :] -= A[i, :] * C[i, :]

            Cf[i, :] /= Bf[i, :]
            Bf[i+1, :] -= Af[i, :] * Cf[i, :]

        B = 1.0 / B
        Bf = 1.0 / Bf

    # Prepare RHS
    RHSTemp = np.zeros((N, N, n_types))
    for i in range(n_types):
        RHSTemp[:, :, i] = matrix.RHS[i]

    RHSDiag = full_diag(RHSTemp, 0)

    n_diags = 1
    while True:
        next_diag = full_diag(RHSTemp, n_diags)
        prev_diag = full_diag(RHSTemp, -n_diags)

        if (np.any(next_diag) or np.any(prev_diag)) and not (2*n_diags-1 > N):
            n_diags += 1
            RHSDiag = np.hstack([prev_diag, RHSDiag, next_diag])
        else:
            break

    RHSTempf = np.zeros((N, N, n_types))
    for i in range(n_types):
        RHSTempf[:, :, i] = matrix.fRHS[i]

    RHSDiagf = full_diag(RHSTempf, 0)

    n_diagsf = 1
    while True:
        next_diag = full_diag(RHSTempf, n_diagsf)
        prev_diag = full_diag(RHSTempf, -n_diagsf)

        if (np.any(next_diag) or np.any(prev_diag)) and not (2*n_diagsf-1 > N):
            n_diagsf += 1
            RHSDiagf = np.hstack([prev_diag, RHSDiagf, next_diag])
        else:
            break

    # Store outputs in matrix
    matrix.periodic = is_periodic
    matrix.A = A
    matrix.B = B
    matrix.C = C
    matrix.D = D
    matrix.Af = Af
    matrix.Bf = Bf
    matrix.Cf = Cf
    matrix.Df = Df
    matrix.R = RHSDiag
    matrix.nRHS = n_diags
    matrix.Rf = RHSDiagf
    matrix.nRHSf = n_diagsf
    matrix.nTypes = n_types

def full_diag(M, k):
    N = M.shape[0]
    D = np.zeros((N, 1, M.shape[2]))
    for i in range(M.shape[2]):
        D[:, 0, i] = np.diag(np.roll(M[:, :, i], -k, axis=1))
    return D
