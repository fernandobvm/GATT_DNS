import os
import numpy as np
from scipy.sparse import diags, dia_matrix, isspmatrix, isspmatrix_dia, issparse, lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from source.library import *

#Da minha cabeça
class Dimension:
    def __init__(self):
        self.blocks = []
        self.types = []
        self.LHS = []
        self.RHS = []
        self.fLHS = []
        self.fRHS = []

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

class Matrices:
    def __init__(self, mesh, domain, boundary, numMethods):
        self.mesh = mesh
        self.domain = domain
        self.boundary = boundary
        self.numMethods = numMethods

        self.x = Dimension()
        self.y = Dimension()
        self.z = Dimension()

        self.neumman2_coeffs = None
        self.neumman_coeffs = None
        
        
        self.makeMatrices()

    def prepareThomas(self, matrix):
        """
        Prepares the matrices for the Fortran solver by precomputing certain results that would be repeated at runtime.
        """
        # Get sizes
        isPeriodic = matrix.LHS[0][-1, 0] != 0
        nTypes = np.max(matrix.types)
        N = matrix.LHS[0].shape[0]

        # Prepare LHS
        A = np.zeros((N - 1, nTypes))
        B = np.zeros((N, nTypes))
        C = np.zeros((N - 1, nTypes))
        D = np.zeros((N, nTypes))

        A1 = np.zeros((1, nTypes))
        Cn = np.zeros((1, nTypes))

        Af = np.zeros((N - 1, nTypes))
        Bf = np.zeros((N, nTypes))
        Cf = np.zeros((N - 1, nTypes))
        Df = np.zeros((N, nTypes))

        A1f = np.zeros((1, nTypes))
        Cnf = np.zeros((1, nTypes))

        for i in range(nTypes):
            A[:, i] = matrix.LHS[i].diagonal(k=-1)
            B[:, i] = matrix.LHS[i].diagonal()
            C[:, i] = matrix.LHS[i].diagonal(k=1)

            Af[:, i] = matrix.fLHS[i].diagonal(k=-1)
            Bf[:, i] = matrix.fLHS[i].diagonal()
            Cf[:, i] = matrix.fLHS[i].diagonal(k=1)

        if isPeriodic:
            for i in range(nTypes):
                A1[0, i] = matrix.LHS[i][0, -1]
                Cn[0, i] = matrix.LHS[i][-1, 0]

                A1f[0, i] = matrix.fLHS[i][0, -1]
                Cnf[0, i] = matrix.fLHS[i][-1, 0]

            A = A[1:, :]
            B = B[1:, :]
            C = C[1:, :]
            Af = Af[1:, :]
            Bf = Bf[1:, :]
            Cf = Cf[1:, :]

            for i in range(N - 2):
                C[i, :] /= B[i, :]
                B[i + 1, :] -= A[i, :] * C[i, :]

                Cf[i, :] /= Bf[i, :]
                Bf[i + 1, :] -= Af[i, :] * Cf[i, :]

            B = 1.0 / B
            Bf = 1.0 / Bf

            # A1 and Cn will be stored as the first elements of A and C
            A = np.vstack([A1, A])
            Af = np.vstack([A1f, Af])
            C = np.vstack([Cn, C])
            Cf = np.vstack([Cnf, Cf])

            for i in range(nTypes):
                Dtemp = np.linalg.inv(matrix.LHS[i])
                D[:, i] = Dtemp[0, :]

                Dtemp = np.linalg.inv(matrix.fLHS[i])
                Df[:, i] = Dtemp[0, :]

        else:
            A1[:] = 0
            Cn[:] = 0
            A1f[:] = 0
            Cnf[:] = 0

            for i in range(N - 1):
                C[i, :] /= B[i, :]
                B[i + 1, :] -= A[i, :] * C[i, :]

                Cf[i, :] /= Bf[i, :]
                Bf[i + 1, :] -= Af[i, :] * Cf[i, :]

            B = 1.0 / B
            Bf = 1.0 / Bf


        done = False
        RHSTemp = np.zeros((N, N, nTypes))

        for i in range(nTypes):
            RHSTemp[:, :, i] = matrix.RHS[i].todense()

        RHSDiag = self.fullDiag(RHSTemp, 0)

        nDiags = 1
        while not done:
            nextDiag = self.fullDiag(RHSTemp, nDiags)
            prevDiag = self.fullDiag(RHSTemp, -nDiags)

            if np.any(nextDiag) or np.any(prevDiag) and (2 * nDiags - 1 <= N):
                nDiags += 1
                RHSDiag = np.hstack([prevDiag, RHSDiag, nextDiag])
            else:
                done = True

        done = False
        RHSTemp = np.zeros((N, N, nTypes))

        for i in range(nTypes):
            RHSTemp[:, :, i] = matrix.fRHS[i].todense()

        RHSDiagf = self.fullDiag(RHSTemp, 0)

        nDiagsf = 1
        while not done:
            nextDiag = self.fullDiag(RHSTemp, nDiagsf)
            prevDiag = self.fullDiag(RHSTemp, -nDiagsf)

            if (np.any(nextDiag) or np.any(prevDiag)) and (2 * nDiagsf - 1 <= N):
                nDiagsf += 1
                RHSDiagf = np.hstack([prevDiag, RHSDiagf, nextDiag])
            else:
                done = True

        # Store outputs
        matrix.periodic = isPeriodic
        matrix.A = A
        matrix.B = B
        matrix.C = C
        matrix.D = D
        matrix.Af = Af
        matrix.Bf = Bf
        matrix.Cf = Cf
        matrix.Df = Df
        matrix.R = RHSDiag
        matrix.nRHS = nDiags
        matrix.Rf = RHSDiagf
        matrix.nRHSf = nDiagsf
        matrix.nTypes = nTypes

        return matrix
        # Prepare RHS

    def fullDiag(self, M, k):
        D = np.zeros((M.shape[0], 1, M.shape[2]))
        for i in range(M.shape[2]):
            D[:, 0, i] = np.diag(np.roll(M[:, :, i], shift=-k, axis=1))
        return D

    def __findDerivativeRegions(self, boundary, mesh):

        # In X
        flow_region = boundary.flow_region.transpose(1, 0, 2)  # Permute equivalent
        
        derivStartsX = []
        derivEndsX = []

        # Find all different types of geometries
        C = np.vstack([np.unique(flow_region[:, k, :], axis=0) for k in range(mesh.nz)])
        C = np.unique(C, axis=0)

        # Identify where these regions are present
        typeMapX = np.zeros((mesh.ny, mesh.nz), dtype=int)
        for j in range(mesh.ny):
            for k in range(mesh.nz):
                typeMapX[j, k] = np.where(np.all(flow_region[j, k, :] == C, axis=1))[0][0] + 1
        for i in range(C.shape[0]):
            if np.any(C[i, :] != 0):  # Only record regions with flow
                if mesh.x.periodic:
                    derivStartsX.append(np.unique(np.where(np.diff(C[i, :]) == 1)[0]))
                    derivEndsX.append(np.unique(np.where(np.diff(C[i, :]) == -1)[0] + 1))
                else:
                    derivStartsX.append(np.unique(np.concatenate(([0], np.where(np.diff(C[i, :]) == 1)[0]))))
                    derivEndsX.append(np.unique(np.concatenate((np.where(np.diff(C[i, :]) == -1)[0] + 1, [mesh.nx - 1]))))
            else:
                typeMapX[typeMapX == i + 1] = 0
                typeMapX[typeMapX > i + 1] = typeMapX[typeMapX > 1] - 1
        typeMapX[typeMapX == 0] = np.max(typeMapX)

        if hasattr(mesh.x, 'breakPoint'):
            if (np.ndim(mesh.x.breakPoint) == 1):
                tam = 1
            else:
                tam = mesh.x.breakPoint.shape[0]

            for j in range(tam):
                if np.ndim(mesh.x.breakPoint) == 1:
                    ind = mesh.x.breakPoint[1:5]
                else:
                    ind = mesh.x.breakPoint[j, 1:5]  # Adjust indices for Python
                types = np.unique(typeMapX[ind[0]:ind[1]+1, ind[2]:ind[3]+1])
                for type_i in types:
                    typeN = np.max(typeMapX) + 1
                    derivStartsX.append(derivStartsX[type_i - 1][:])
                    derivEndsX.append(derivEndsX[type_i - 1][:])

                    if np.ndim(mesh.x.breakPoint) == 1:
                        derivStartsX[typeN - 1] = np.hstack((derivStartsX[typeN - 1], mesh.x.breakPoint[0] + 1))
                        derivEndsX[typeN - 1] =np.hstack((derivEndsX[typeN - 1], mesh.x.breakPoint[0]))
                    else:
                        derivStartsX[typeN - 1] = np.hstack((derivStartsX[typeN - 1], mesh.x.breakPoint[j, 0] + 1))
                        derivEndsX[typeN - 1] =np.hstack((derivEndsX[typeN - 1], mesh.x.breakPoint[j, 0]))  
                    

                    typeMapTemp = typeMapX[ind[0]:ind[1]+1, ind[2]:ind[3]+1]
                    typeMapTemp[typeMapTemp == type_i] = typeN
                    typeMapX[ind[0]:ind[1]+1, ind[2]:ind[3]+1] = typeMapTemp

        # In Y
        flow_region = boundary.flow_region.T

        derivStartsY = []
        derivEndsY = []

        # Find all different types of geometries
        C = np.vstack([np.unique(flow_region[:, :, k], axis=0) for k in range(mesh.nz)])
        C = np.unique(C, axis=0)

        # Identify where these regions are present
        typeMapY = np.zeros((mesh.nx, mesh.nz), dtype=int)
        for i in range(mesh.nx):
            for k in range(mesh.nz):
                typeMapY[i, k] = np.where(np.all(flow_region[i, :, k] == C, axis=1))[0][0] + 1

        for i in range(C.shape[0]):
            if np.any(C[i, :] != 0):
                if mesh.y.periodic:
                    derivStartsY.append(np.unique(np.where(np.diff(C[i, :]) == 1)[0]))
                    derivEndsY.append(np.unique(np.where(np.diff(C[i, :]) == -1)[0] + 1))
                else:
                    derivStartsY.append(np.unique(np.concatenate(([0], np.where(np.diff(C[i, :]) == 1)[0]))))
                    derivEndsY.append(np.unique(np.concatenate((np.where(np.diff(C[i, :]) == -1)[0] + 1, [mesh.ny - 1]))))
            else:
                typeMapY[typeMapY == i + 1] = 0
                typeMapY[typeMapY > i + 1] -= 1

        typeMapY[typeMapY == 0] = np.max(typeMapY)

        if hasattr(mesh.y, 'breakPoint'):
            if (np.ndim(mesh.y.breakPoint) == 1):
                tam = 1
            else:
                tam = mesh.y.breakPoint.shape[0]

            for j in range(tam):
                if np.ndim(mesh.y.breakPoint) == 1:
                    ind = mesh.y.breakPoint[1:5] 
                else:
                    ind = mesh.y.breakPoint[j, 1:5] 
                types = np.unique(typeMapY[ind[0]:ind[1]+1, ind[2]:ind[3]+1])
                for type_i in types:
                    typeN = np.max(typeMapY) + 1
                    derivStartsY.append(derivStartsY[type_i - 1][:])
                    derivEndsY.append(derivEndsY[type_i - 1][:])

                    if np.ndim(mesh.y.breakPoint) == 1:
                        derivStartsY[typeN - 1] = np.hstack((derivStartsY[typeN - 1], mesh.y.breakPoint[0] + 1))
                        derivEndsY[typeN - 1] = np.hstack((derivEndsY[typeN - 1]), mesh.y.breakPoint[0])
                    else:
                        derivStartsY[typeN - 1] = np.hstack((derivStartsY[typeN - 1], mesh.y.breakPoint[j, 0] + 1))
                        derivEndsY[typeN - 1] = np.hstack((derivEndsY[typeN - 1]), mesh.y.breakPoint[j, 0])

                    typeMapTemp = typeMapY[ind[0]:ind[1]+1, ind[2]:ind[3]+1]
                    typeMapTemp[typeMapTemp == type_i] = typeN
                    typeMapY[ind[0]:ind[1]+1, ind[2]:ind[3]+1] = typeMapTemp

        # In Z
        flow_region = boundary.flow_region.transpose(2, 0, 1)

        derivStartsZ = []
        derivEndsZ = []

        # Find all different types of geometries
        C = np.vstack([np.unique(flow_region[:, :, j], axis=0) for j in range(mesh.ny)])
        C = np.unique(C, axis=0)

        # Identify where these regions are present
        typeMapZ = np.zeros((mesh.nx, mesh.ny), dtype=int)
        for i in range(mesh.nx):
            for j in range(mesh.ny):
                typeMapZ[i, j] = np.where(np.all(flow_region[i, :, j] == C, axis=1))[0][0] + 1

        for i in range(C.shape[0]):
            if np.any(C[i, :] != 0):
                if mesh.z.periodic:
                    derivStartsZ.append(np.unique(np.where(np.diff(C[i, :]) == 1)[0]))
                    derivEndsZ.append(np.unique(np.where(np.diff(C[i, :]) == -1)[0] + 1))
                else:
                    derivStartsZ.append(np.unique(np.concatenate(([0], np.where(np.diff(C[i, :]) == 1)[0]))))
                    derivEndsZ.append(np.unique(np.concatenate((np.where(np.diff(C[i, :]) == -1)[0] + 1, [mesh.nz - 1]))))
            else:
                typeMapZ[typeMapZ == i + 1] = 0
                typeMapZ[typeMapZ > i + 1] -= 1

        typeMapZ[typeMapZ == 0] = np.max(typeMapZ)

        if hasattr(mesh.z, 'breakPoint'):
            if (np.ndim(mesh.z.breakPoint) == 1):
                tam = 1
            else:
                tam = mesh.z.breakPoint.shape[0]

            for j in range(tam):
                if np.ndim(mesh.z.breakPoint) == 1:
                    ind = mesh.z.breakPoint[1:5] 
                else:
                    ind = mesh.z.breakPoint[j, 1:5] 
                types = np.unique(typeMapZ[ind[0]:ind[1]+1, ind[2]:ind[3]+1])
                for type_i in types:
                    typeN = np.max(typeMapZ) + 1
                    derivStartsZ.append(derivStartsZ[type_i - 1][:])
                    derivEndsZ.append(derivEndsZ[type_i - 1][:])

                    if np.ndim(mesh.y.breakPoint) == 1:
                        derivStartsZ[typeN - 1] = np.hstack((derivStartsZ[typeN - 1], mesh.z.breakPoint[0] + 1))
                        derivEndsZ[typeN - 1] = np.hstack((derivEndsZ[typeN - 1]), mesh.z.breakPoint[0])
                    else:
                        derivStartsZ[typeN - 1] = np.hstack((derivStartsZ[typeN - 1], mesh.z.breakPoint[j, 0] + 1))
                        derivEndsZ[typeN - 1] = np.hstack((derivEndsZ[typeN - 1]), mesh.z.breakPoint[j, 0])


                    typeMapTemp = typeMapZ[ind[0]:ind[1]+1, ind[2]:ind[3]+1]
                    typeMapTemp[typeMapTemp == type_i] = typeN
                    typeMapZ[ind[0]:ind[1]+1, ind[2]:ind[3]+1] = typeMapTemp

        return derivStartsX, derivStartsY, derivStartsZ, derivEndsX, derivEndsY, derivEndsZ, typeMapX, typeMapY, typeMapZ
    
    def makeMatrices(self):

        derivStartsX, derivStartsY, derivStartsZ, derivEndsX, derivEndsY, derivEndsZ, typeMapX, typeMapY, typeMapZ = self.__findDerivativeRegions(self.boundary, self.mesh)

        # Get finite differences coefficients
        centeredStencilLHS, centeredStencilRHS, decenteredStencilLHS, decenteredStencilRHS = self.__finiteDifferenceCoefficients(self.numMethods.spatial_derivs)
        centeredStencilLHSb, centeredStencilRHSb, decenteredStencilLHSb, decenteredStencilRHSb = self.__finiteDifferenceCoefficients(self.numMethods.spatial_derivs_buffer)
        filterStencilLHS, filterStencilRHS, filterDecenteredStencilLHS, filterDecenteredStencilRHS = self.__spatialFilterCoefficients(self.numMethods.spatial_filter_strength, self.numMethods.filter_borders)

        # Handle Z direction and filter stencil based on mesh size
        if self.mesh.nz > 1 and self.mesh.nz < 2 * len(filterStencilRHS) - 1:
            filterStencilLHSz = 1
            filterStencilRHSz = 1
            filterDecenteredStencilLHSz = 1
            filterDecenteredStencilRHSz = 1
        else:
            filterStencilLHSz = filterStencilLHS
            filterStencilRHSz = filterStencilRHS
            filterDecenteredStencilLHSz = filterDecenteredStencilLHS
            filterDecenteredStencilRHSz = filterDecenteredStencilRHS

        # Switch to spectral mode if mesh in Z has exactly 4 nodes and no boundaries
        if self.mesh.nz == 4 and len(derivStartsZ) == 1 and not derivStartsZ[0] and not derivEndsZ[0]:
            centeredStencilLHSz = 1
            centeredStencilRHSz = [0, 3.14159 / 4]  # Pi/4
            centeredStencilLHSbz = centeredStencilLHSz
            centeredStencilRHSbz = centeredStencilRHSz
        else:
            centeredStencilLHSz = centeredStencilLHS
            centeredStencilRHSz = centeredStencilRHS
            centeredStencilLHSbz = centeredStencilLHSb
            centeredStencilRHSbz = centeredStencilRHSb

        # Make matrices for each direction
        LHSx, RHSx = self.__makeMatricesEachDirection(centeredStencilLHS, centeredStencilRHS, decenteredStencilLHS, decenteredStencilRHS, derivStartsX, derivEndsX, self.mesh.nx, None)
        LHSy, RHSy = self.__makeMatricesEachDirection(centeredStencilLHS, centeredStencilRHS, decenteredStencilLHS, decenteredStencilRHS, derivStartsY, derivEndsY, self.mesh.ny, None)
        LHSz, RHSz = self.__makeMatricesEachDirection(centeredStencilLHSz, centeredStencilRHSz, decenteredStencilLHS, decenteredStencilRHS, derivStartsZ, derivEndsZ, self.mesh.nz, None)

        LHSxb, RHSxb = self.__makeMatricesEachDirection(centeredStencilLHSb, centeredStencilRHSb, decenteredStencilLHSb, decenteredStencilRHSb, derivStartsX, derivEndsX, self.mesh.nx, self.mesh.x)
        LHSyb, RHSyb = self.__makeMatricesEachDirection(centeredStencilLHSb, centeredStencilRHSb, decenteredStencilLHSb, decenteredStencilRHSb, derivStartsY, derivEndsY, self.mesh.ny, self.mesh.y)
        LHSzb, RHSzb = self.__makeMatricesEachDirection(centeredStencilLHSbz, centeredStencilRHSbz, decenteredStencilLHSb, decenteredStencilRHSb, derivStartsZ, derivEndsZ, self.mesh.nz, self.mesh.z)

        fLHSx, fRHSx = self.__makeMatricesEachDirection(filterStencilLHS, filterStencilRHS, filterDecenteredStencilLHS, filterDecenteredStencilRHS, derivStartsX, derivEndsX, self.mesh.nx, None)
        fLHSy, fRHSy = self.__makeMatricesEachDirection(filterStencilLHS, filterStencilRHS, filterDecenteredStencilLHS, filterDecenteredStencilRHS, derivStartsY, derivEndsY, self.mesh.ny, None)
        fLHSz, fRHSz = self.__makeMatricesEachDirection(filterStencilLHSz, filterStencilRHSz, filterDecenteredStencilLHSz, filterDecenteredStencilRHSz, derivStartsZ, derivEndsZ, self.mesh.nz, None)

        # Add buffer zones to derivative matrices
        if not hasattr(self.numMethods, 'changeOrderX') or self.numMethods.changeOrderX:
            LHSx, RHSx = self.__addBufferToMatrix(LHSx, LHSxb, RHSx, RHSxb, self.mesh.nx, self.mesh.x)
        if not hasattr(self.numMethods, 'changeOrderY') or self.numMethods.changeOrderY:
            LHSy, RHSy = self.__addBufferToMatrix(LHSy, LHSyb, RHSy, RHSyb, self.mesh.ny, self.mesh.y)
        if not hasattr(self.numMethods, 'changeOrderZ') or self.numMethods.changeOrderZ:
            LHSz, RHSz = self.__addBufferToMatrix(LHSz, LHSzb, RHSz, RHSzb, self.mesh.nz, self.mesh.z)

        # Choose spatial derivative method for the metric
        if not hasattr(self.numMethods, 'metricMethod'):
            self.numMethods.metricMethod = 'SL4'
        elif not self.numMethods.metricMethod:
            self.numMethods.metricMethod = self.numMethods.spatialDerivs

        if self.mesh.nz == 4:
            self.numMethods.metricMethodZ = 'SL4'
        else:
            self.numMethods.metricMethodZ = self.numMethods.metricMethod

        LHSx = self.__applyMetric(LHSx, self.mesh.X, self.mesh.x, self.domain.xf, self.numMethods.metricMethod)
        LHSy = self.__applyMetric(LHSy, self.mesh.Y, self.mesh.y, self.domain.yf, self.numMethods.metricMethod)
        LHSz = self.__applyMetric(LHSz, self.mesh.Z, self.mesh.z, self.domain.zf, self.numMethods.metricMethodZ)

        # Remove ends of filters if needed
        fLHSx, fRHSx = self.__removeFilterEnds(self.numMethods, fLHSx, fRHSx, 'X')
        fLHSy, fRHSy = self.__removeFilterEnds(self.numMethods, fLHSy, fRHSy, 'Y')
        fLHSz, fRHSz = self.__removeFilterEnds(self.numMethods, fLHSz, fRHSz, 'Z')

        # Save to output structure
        self.x.types = typeMapX
        self.y.types = typeMapY
        self.z.types = typeMapZ

        self.x.LHS = LHSx
        self.x.RHS = RHSx
        self.y.LHS = LHSy
        self.y.RHS = RHSy
        self.z.LHS = LHSz
        self.z.RHS = RHSz

        self.x.fLHS = fLHSx
        self.x.fRHS = fRHSx
        self.y.fLHS = fLHSy
        self.y.fRHS = fRHSy
        self.z.fLHS = fLHSz
        self.z.fRHS = fRHSz

    def __finiteDifferenceCoefficients(self, method):

        # Centered and decentered stencils placeholders
        centeredStencilLHS = []
        centeredStencilRHS = []
        decenteredStencilLHS = []
        decenteredStencilRHS = []

        if method == 'SL6':
            w = 1.8

            # Center
            matriz_a = np.array([
                [1, 1, 1, -2],
                [1, 4, 9, -6],
                [1, 16, 81, -10],
                [np.sin(w), np.sin(2*w)/2, np.sin(3*w)/3, -2*w*np.cos(w)]
            ])
            matriz_b = np.array([1, 0, 0, w])

            coeffs = np.linalg.solve(matriz_a, matriz_b)
            a = coeffs[0] / 2
            b = coeffs[1] / 4
            c = coeffs[2] / 6
            alpha = coeffs[3]

            centeredStencilLHS = [1, alpha]
            centeredStencilRHS = [0, a, b, c]

            # Border - First stage
            matriz_a = np.array([
                [1, 1, 1, 1, 1, 1, 0],
                [0, 1, 2, 3, 4, 5, -1],
                [0, 1, 4, 9, 16, 25, -2],
                [0, 1, 8, 27, 64, 125, -3],
                [0, 1, 16, 81, 256, 625, -4],
                [0, 1, 32, 243, 1024, 3125, -5],
                [0, 1, 64, 729, 4096, 15625, -6]
            ])
            matriz_b = np.array([0, 1, 0, 0, 0, 0, 0])

            coeffs = np.linalg.solve(matriz_a, matriz_b)
            a, b, c, d, e, f, alpha = coeffs

            matriz_lhs_aux = [1, alpha, 0, 0]
            matriz_rhs_aux = [a, b, c, d, e, f]

            # Second stage
            matriz_a = np.array([[-1, 2], [-1, 6]])
            matriz_b = np.array([-1, 0])

            coeffs = np.linalg.solve(matriz_a, matriz_b)
            a = coeffs[0] / 2
            alpha = coeffs[1]

            matriz_lhs_aux2 = [alpha, 1, alpha, 0]
            matriz_rhs_aux2 = [-a, 0, a, 0, 0, 0]

            # Third stage (same as centered SL4)
            matriz_a = np.array([
                [1, 0, -2/3],
                [0, 1, -4/3],
                [np.sin(w), np.sin(2*w)/2, -2*w*np.cos(w)]
            ])
            matriz_b = np.array([4/3, -1/3, w])

            coeffs = np.linalg.solve(matriz_a, matriz_b)
            a = coeffs[0] / 2
            b = coeffs[1] / 4
            alpha = coeffs[2]

            matriz_lhs_aux3 = [0, alpha, 1, alpha]
            matriz_rhs_aux3 = [-b, -a, 0, a, b, 0]

            decenteredStencilLHS = [matriz_lhs_aux, matriz_lhs_aux2, matriz_lhs_aux3]
            decenteredStencilRHS = [matriz_rhs_aux, matriz_rhs_aux2, matriz_rhs_aux3]

        elif method == 'SL6O3':
            coeffs = [0.392465753424658, 1.565410958904110, 0.237260273972603, -0.017739726027397]
            alpha, a, b, c = coeffs
            centeredStencilLHS = [1, alpha]
            centeredStencilRHS = [0, a/2, b/4, c/6]

            coeffs_P3 = [0.350978473581213, 1.567318982387476, 0.134637964774951]
            alpha_P3, a_P3, b_P3 = coeffs_P3

            decenteredStencilLHS = [
                [1, (3*np.pi + 40)/(3*np.pi + 8), 0, 0],
                [1/4, 1, 1/4, 0],
                [0, alpha_P3, 1, alpha_P3]
            ]
            decenteredStencilRHS = [
                [-(13*np.pi + 56)/(2*(3*np.pi + 8)), (15*np.pi + 8)/(2*(3*np.pi + 8)), -(3*np.pi - 56)/(2*(3*np.pi + 8)), (np.pi - 8)/(2*(3*np.pi + 8)), 0],
                [-3/4, 0, 3/4, 0, 0],
                [-b_P3/4, -a_P3/2, 0, a_P3/2, b_P3/4]
            ]

        elif method == 'SL4':
            w = 1.8
            matriz_a = np.array([
                [1, 0, -2/3],
                [0, 1, -4/3],
                [np.sin(w), np.sin(2*w)/2, -2*w*np.cos(w)]
            ])
            matriz_b = np.array([4/3, -1/3, w])

            coeffs = np.linalg.solve(matriz_a, matriz_b)
            a = coeffs[0] / 2
            b = coeffs[1] / 4
            alpha = coeffs[2]

            centeredStencilLHS = [1, alpha]
            centeredStencilRHS = [0, a, b]

            decenteredStencilLHS = [
                [1, 3, 0],
                [1/4, 1, 1/4]
            ]
            decenteredStencilRHS = [
                [-17/6, 3/2, 3/2, -1/6],
                [-3/4, 0, 3/4, 0]
            ]

        elif method == 'EX2':
            centeredStencilLHS = 1
            centeredStencilRHS = [0, 1/2]

            decenteredStencilLHS = 1
            decenteredStencilRHS = [-3/2, 2, -1/2]

        elif method == 'EX4':
            centeredStencilLHS = 1
            centeredStencilRHS = [0, 2/3, -1/12]

            decenteredStencilLHS = 1
            decenteredStencilRHS = [
                [-25/12, 4, -3, 4/3, -1/4],
                [-1/2, 0, 1/2, 0, 0]
            ]

        else:
            raise ValueError(f'Finite differences method not implemented: {method}. Check finiteDifferenceCoefficients for available methods')

        def to_numpy_vector(value):
            if np.isscalar(value):
                return np.array([value])
            else:
                return np.array(value)

        return [
            to_numpy_vector(centeredStencilLHS),
            to_numpy_vector(centeredStencilRHS),
            to_numpy_vector(decenteredStencilLHS),
            to_numpy_vector(decenteredStencilRHS)
        ]

    
    def __spatialFilterCoefficients(self, alpha, filterBorders):
        # Coefficients for the 10th order spatial filter from Gaitonde 1998 for a given alpha
        
        filterStencilLHS = [1, alpha]
        
        filterStencilRHS = [
            (193 + 126 * alpha) / 256,
            (105 + 302 * alpha) / 512,
            (-15 + 30 * alpha) / 128,
            (45 - 90 * alpha) / 1024,
            (-5 + 10 * alpha) / 512,
            (1 - 2 * alpha) / 1024
        ]
        
        if isinstance(filterBorders, bool):
            if filterBorders:
                filterBorders = 'decentered'
            else:
                filterBorders = 'off'
        
        if filterBorders == 'decentered':
            filterDecenteredStencilLHS = [1, alpha]
            a_bound = [[0 for _ in range(5)] for _ in range(11)]  # 11x5 matrix
            
            a_bound[0][4] = (-1 + 2 * alpha) / 1024
            a_bound[1][4] = (5 - 10 * alpha) / 512
            a_bound[2][4] = (-45 + 90 * alpha) / 1024
            a_bound[3][4] = (15 + 98 * alpha) / 128
            a_bound[4][4] = (407 + 210 * alpha) / 512
            a_bound[5][4] = (63 + 130 * alpha) / 256
            a_bound[6][4] = (-105 + 210 * alpha) / 512
            a_bound[7][4] = (15 - 30 * alpha) / 128
            a_bound[8][4] = (-45 + 90 * alpha) / 1024
            a_bound[9][4] = (5 - 10 * alpha) / 512
            a_bound[10][4] = (-1 + 2 * alpha) / 1024

            # Continue filling a_bound based on the Matlab pattern
            a_bound[0][3] = (1 - 2 * alpha) / 1024
            a_bound[1][3] = (-5 + 10 * alpha) / 512
            a_bound[2][3] = (45 + 934 * alpha) / 1024
            a_bound[3][3] = (113 + 30 * alpha) / 128
            a_bound[4][3] = (105 + 302 * alpha) / 512
            a_bound[5][3] = (-63 + 126 * alpha) / 256
            a_bound[6][3] = (105 - 210 * alpha) / 512
            a_bound[7][3] = (-15 + 30 * alpha) / 128
            a_bound[8][3] = (45 - 90 * alpha) / 1024
            a_bound[9][3] = (-5 + 10 * alpha) / 512
            a_bound[10][3] = (1 - 2 * alpha) / 1024

            # Continue similarly for columns 1, 2
            # ...
            
            filterDecenteredStencilRHS = list(map(list, zip(*a_bound)))  # Transpose matrix a_bound
            
        elif filterBorders == 'reducedOrder':
            filterDecenteredStencilLHS = 1
            
            F2 = np.array([(1 / 2 + alpha), (1 / 2 + alpha)]) / np.array([1, 2])
            F4 = np.array([(5 / 8 + 3 / 4 * alpha), (1 / 2 + alpha), (-1 / 8 + 1 / 4 * alpha)]) / np.array([1, 2, 2])
            F6 = np.array([(11 / 16 + 5 / 8 * alpha), (15 / 32 + 17 / 16 * alpha), (-3 / 16 + 3 / 8 * alpha), (1 / 32 - 1 / 16 * alpha)]) / np.array([1, 2, 2, 2])
            F8 = np.array([(93 / 128 + 70 / 128 * alpha), (7 / 16 + 18 / 16 * alpha), (-7 / 32 + 14 / 32 * alpha), (1 / 16 - 1 / 8 * alpha), (-1 / 128 + 1 / 64 * alpha)]) / np.array([1, 2, 2, 2, 2])
            
            filterDecenteredStencilRHS = [[0] * 9 for _ in range(5)]  # 5x9 matrix
            
            filterDecenteredStencilRHS[0][0] = 1
            filterDecenteredStencilRHS[1][0:3] = [F2[1], F2[0], F2[1]]
            filterDecenteredStencilRHS[2][0:5] = [F4[2], F4[1], F4[0], F4[1], F4[2]]
            filterDecenteredStencilRHS[3][0:7] = [F6[3], F6[2], F6[1], F6[0], F6[1], F6[2], F6[3]]
            filterDecenteredStencilRHS[4][0:9] = [F8[4], F8[3], F8[2], F8[1], F8[0], F8[1], F8[2], F8[3], F8[4]]
        
        elif filterBorders == 'off':
            filterDecenteredStencilLHS = np.eye(5)
            filterDecenteredStencilRHS = np.eye(5)
        
        return [filterStencilLHS, filterStencilRHS, filterDecenteredStencilLHS, filterDecenteredStencilRHS]
    
    def __makeMatricesEachDirection2(self, centeredStencilLHS, centeredStencilRHS, decenteredStencilLHS, decenteredStencilRHS, derivStarts, derivEnds, n, bufferInfo):
        
        if n != 1 and n < 2 * len(centeredStencilRHS) - 1:
            raise ValueError(f'Mesh is not large enough for one of the stencils. It has {n} nodes but the stencil needs at least {2 * len(centeredStencilRHS) - 1}.')
        
        nTypes = len(derivStarts)
        
        LHS = [None] * nTypes
        RHS = [None] * nTypes
        
        if n == 1:
            if centeredStencilRHS[0] == 0:
                for i in range(nTypes):
                    LHS[i] = 1
                    RHS[i] = 0
            else:
                for i in range(nTypes):
                    LHS[i] = 1
                    RHS[i] = 1
            return LHS, RHS
        
        LHS_base = diags([centeredStencilLHS[0]] * n, 0).toarray()
        RHS_base = diags([centeredStencilRHS[0]] * n, 0).toarray()
        
        invertStencil = -1 if centeredStencilRHS[0] == 0 else 1
        
        for i in range(1, len(centeredStencilLHS)):
            inds = np.arange(n) + i
            inds = np.mod(inds, n)
            LHS_base[np.arange(n), inds] = centeredStencilLHS[i]
            inds = np.arange(n) - i
            inds = np.mod(inds, n)
            LHS_base[np.arange(n), inds] = centeredStencilLHS[i]
        
        for i in range(1, len(centeredStencilRHS)):
            inds = np.arange(n) + i
            inds = np.mod(inds, n)
            RHS_base[np.arange(n), inds] = centeredStencilRHS[i]
            inds = np.arange(n) - i
            inds = np.mod(inds, n)
            RHS_base[np.arange(n), inds] = invertStencil * centeredStencilRHS[i]
        
        if bufferInfo:
            if hasattr(bufferInfo.buffer_i, 'upwind') and bufferInfo.buffer_i.upwind:
                for i in range(nTypes):
                    derivStarts[i] = sorted(set(derivStarts[i] + list(range(1, bufferInfo.buffer_i.n + 1))), reverse=True)
            if hasattr(bufferInfo.buffer_f, 'upwind') and bufferInfo.buffer_f.upwind:
                for i in range(nTypes):
                    derivEnds[i] = sorted(set(derivEnds[i] + list(range(n - bufferInfo.buffer_f.n + 1, n + 1))))
        
        mLHS, nLHS = get_array_dimensions(decenteredStencilLHS)
        mRHS, nRHS = get_array_dimensions(decenteredStencilRHS)
        
        for i in range(nTypes):
            LHS_temp = LHS_base.copy()
            RHS_temp = RHS_base.copy()
            
            for ind_start in derivStarts[i]:
                LHS_temp[ind_start:ind_start + mLHS] = 0
                RHS_temp[ind_start:ind_start + mRHS] = 0
                LHS_temp[ind_start:ind_start + mLHS, ind_start:ind_start + nLHS] = decenteredStencilLHS
                RHS_temp[ind_start:ind_start + mRHS, ind_start:ind_start + nRHS] = decenteredStencilRHS
            
            for ind_end in derivEnds[i]:
                LHS_temp[ind_end - mLHS + 1:ind_end + 1] = 0
                RHS_temp[ind_end - mRHS + 1:ind_end + 1] = 0
                LHS_temp[ind_end - mLHS + 1:ind_end + 1, ind_end - nLHS + 1:ind_end + 1] = np.flipud(np.fliplr(decenteredStencilLHS))
                RHS_temp[ind_end - mRHS + 1:ind_end + 1, ind_end - nRHS + 1:ind_end + 1] = invertStencil * np.flipud(np.fliplr(decenteredStencilRHS))
            
            LHS[i] = lil_matrix(LHS_temp)
            RHS[i] = lil_matrix(RHS_temp)
        
        return LHS, RHS
    
    
    def __makeMatricesEachDirection(self, centeredStencilLHS, centeredStencilRHS, decenteredStencilLHS, decenteredStencilRHS, derivStarts, derivEnds, n, bufferInfo):
        # Check if the mesh is large enough for the stencil
        if n != 1 and n < 2 * len(centeredStencilRHS) - 1:
            raise ValueError(f"Mesh is not large enough for one of the stencils. It has {n} nodes but the stencil needs at least {2 * len(centeredStencilRHS) - 1}.")

        nTypes = len(derivStarts)

        LHS = [None] * nTypes
        RHS = [None] * nTypes

        if n == 1:  # If single point in this direction, set derivative to zero and filter to one
            if centeredStencilRHS[0] == 0:
                for i in range(nTypes):
                    LHS[i] = 1
                    RHS[i] = 0
            else:
                for i in range(nTypes):
                    LHS[i] = 1
                    RHS[i] = 1
            return LHS, RHS

        # Create the base stencils
        LHS_base = np.diag(centeredStencilLHS[0] * np.ones(n))
        RHS_base = np.diag(centeredStencilRHS[0] * np.ones(n))

        invertStencil = -1 if centeredStencilRHS[0] == 0 else 1

        for i in range(1, len(centeredStencilLHS)):
            row_indices = np.arange(0,n)
            col_indices = np.mod(row_indices + i, n)
            inds = np.ravel_multi_index((col_indices, row_indices), (n, n))

            LHS_base.flat[inds] = centeredStencilLHS[i]

            row_indices = np.arange(0,n)
            col_indices = np.mod(row_indices - i, n)
            inds = np.ravel_multi_index((col_indices, row_indices), (n, n))

            LHS_base.flat[inds] = centeredStencilLHS[i]
            

        for i in range(1, len(centeredStencilRHS)):
            row_indices = np.arange(0,n)
            col_indices = np.mod(row_indices + i, n)
            inds = np.ravel_multi_index((col_indices, row_indices), (n, n))

            RHS_base.flat[inds] = centeredStencilRHS[i]

            row_indices = np.arange(0,n)
            col_indices = np.mod(row_indices - i, n)
            inds = np.ravel_multi_index((col_indices, row_indices), (n, n))

            RHS_base.flat[inds] = invertStencil*centeredStencilRHS[i]

        # Check if this is a buffer zone that needs to be upwind
        # and add that to the list of starts and ends so that the decentered stencil is used
        if bufferInfo is not None:
            if hasattr(bufferInfo.buffer_i, 'upwind') and bufferInfo.buffer_i.upwind:
                for i in range(nTypes):
                    derivStarts[i] = sorted(set(derivStarts[i] + list(range(1, bufferInfo.buffer_i.n + 1))), reverse=True)
            if hasattr(bufferInfo.buffer_f, 'upwind') and bufferInfo.buffer_f.upwind:
                for i in range(nTypes):
                    derivEnds[i] = sorted(set(derivEnds[i] + list(range(n - bufferInfo.buffer_f.n + 1, n + 1))))

        LHS_base = LHS_base.T
        RHS_base = RHS_base.T
        
        # Add startings and endings
        
        mLHS, nLHS = get_array_dimensions(decenteredStencilLHS)
        mRHS, nRHS = get_array_dimensions(decenteredStencilRHS)

        for i in range(nTypes):
            LHS_temp = LHS_base.copy()
            RHS_temp = RHS_base.copy()

            for ind_start in ensure_iterable(derivStarts[i]):
                LHS_temp[ind_start:ind_start + mLHS, :] = 0
                RHS_temp[ind_start:ind_start + mRHS, :] = 0

                #if decenteredStencilLHS.ndim == 1:
                #    decenteredStencilLHS = decenteredStencilLHS[:,None]
                #if decenteredStencilRHS.ndim == 1:
                #    decenteredStencilRHS = decenteredStencilRHS[:,None]

                LHS_temp[ind_start:ind_start + mLHS, ind_start:ind_start + nLHS] = decenteredStencilLHS
                RHS_temp[ind_start:ind_start + mRHS, ind_start:ind_start + nRHS] = decenteredStencilRHS

            for ind_end in ensure_iterable(derivEnds[i]):
                LHS_temp[ind_end - mLHS + 1:ind_end + 1, :] = 0
                RHS_temp[ind_end - mRHS + 1:ind_end + 1, :] = 0

                

                LHS_temp[ind_end - mLHS + 1:ind_end + 1, ind_end - nLHS + 1:ind_end + 1] = safe_flip(decenteredStencilLHS)
                RHS_temp[ind_end - mRHS + 1:ind_end + 1, ind_end - nRHS + 1:ind_end + 1] = invertStencil * safe_flip(decenteredStencilRHS)

            #LHS[i] = diags(LHS_temp.diagonal())
            #RHS[i] = diags(RHS_temp.diagonal())

            LHS[i] = csr_matrix(LHS_temp)
            RHS[i] = csr_matrix(RHS_temp)


        return LHS, RHS
    
    def __addBufferToMatrix(self, baseMatrixL, bufferMatrixL, baseMatrixR, bufferMatrixR, n, bufferInfo):
        ni = bufferInfo.buffer_i.n
        nf = bufferInfo.buffer_f.n

        # Handling transitions
        nti = round(bufferInfo.buffer_i.transition * ni) if hasattr(bufferInfo.buffer_i, 'transition') else ni
        ntf = round(bufferInfo.buffer_f.transition * nf) if hasattr(bufferInfo.buffer_f, 'transition') else nf

        ni1 = ni - nti
        ni2 = ni - 1
        nf1 = n - nf
        nf2 = nf1 + ntf - 1

        # The buffer zone can be computed by a different type of derivatives. The transition is done smoothly.
        eta = np.ones(n)

        if ni > 0:
            eta[:ni1] = 0
            eta[ni1:ni2+1] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ni2 - ni1 + 1))

        if nf > 0:
            eta[nf2:] = 0
            eta[nf1:nf2+1] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, nf2 - nf1 + 1))

        newMatrixL = []
        newMatrixR = []

        eta = np.sqrt(eta)

        eta = eta.T
        eta = eta[:, np.newaxis]

        # Para cada tipo de derivada, realizar a operação nas matrizes esparsas
        for baseL, bufferL, baseR, bufferR in zip(baseMatrixL, bufferMatrixL, baseMatrixR, bufferMatrixR):
            # Operar diretamente sem conversões desnecessárias
            if issparse(baseL) and issparse(bufferL):
                #resultL = baseL.multiply(eta[:, np.newaxis]) + bufferL.multiply((1 - eta)[:, np.newaxis])
                #resultR = baseR.multiply(eta[:, np.newaxis]) + bufferR.multiply((1 - eta)[:, np.newaxis])

                resultL = csr_matrix(eta).multiply(baseL) + csr_matrix(1 - eta).multiply(bufferL)
                resultR = csr_matrix(eta).multiply(baseR) + csr_matrix(1 - eta).multiply(bufferR)
            else:
                resultL = eta * baseL + (1 - eta) * bufferL
                resultR = eta * baseR + (1 - eta) * bufferR

            # Converter diretamente para DIA com corretos offsets
            dia_resultL = diags(resultL.diagonal(), offsets=0, shape=resultL.shape, format='dia')
            dia_resultR = diags(resultR.diagonal(), offsets=0, shape=resultR.shape, format='dia')

            #newMatrixL.append(dia_resultL)
            #newMatrixR.append(dia_resultR)

            newMatrixL.append(resultL)
            newMatrixR.append(resultR)

        return newMatrixL, newMatrixR

    def __addBufferToMatrix2(self, baseMatrixL, bufferMatrixL, baseMatrixR, bufferMatrixR, n, bufferInfo):
        ni = bufferInfo.buffer_i.n
        nf = bufferInfo.buffer_f.n

        # Handling transitions
        nti = round(bufferInfo.buffer_i.transition * ni) if hasattr(bufferInfo.buffer_i, 'transition') else ni
        ntf = round(bufferInfo.buffer_f.transition * nf) if hasattr(bufferInfo.buffer_f, 'transition') else nf

        ni1 = ni - nti
        ni2 = ni
        nf1 = n - nf
        nf2 = nf1 + ntf

        # Buffer zone computation, with smooth transition
        eta = np.ones(n)

        if ni > 0:
            eta[:ni1] = 0
            eta[ni1:ni2] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ni2 - ni1))

        if nf > 0:
            eta[nf2:] = 0
            eta[nf1:nf2] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, nf2 - nf1))

        newMatrixL = []
        newMatrixR = []

        # Process each derivative type, directly in sparse matrices
        for baseL, bufferL, baseR, bufferR in zip(baseMatrixL, bufferMatrixL, baseMatrixR, bufferMatrixR):
            if isspmatrix_dia(baseL) and isspmatrix_dia(bufferL):
                # Direct operation on sparse diagonal matrices (DIA)
                resultL = dia_matrix((eta[:, np.newaxis] * baseL.data + (1 - eta[:, np.newaxis]) * bufferL.data, baseL.offsets), shape=baseL.shape)
                resultR = dia_matrix((eta[:, np.newaxis] * baseR.data + (1 - eta[:, np.newaxis]) * bufferR.data, baseR.offsets), shape=baseR.shape)
            else:
                # Handle dense or integer matrices
                resultL = eta[:, np.newaxis] * baseL + (1 - eta[:, np.newaxis]) * bufferL
                resultR = eta[:, np.newaxis] * baseR + (1 - eta[:, np.newaxis]) * bufferR

            newMatrixL.append(resultL)
            newMatrixR.append(resultR)

        return newMatrixL, newMatrixR

    
    def __applyMetric(self, LHS, X, meshInfo, xf, method):
        """
        Scale the LHS matrix due to the mesh stretching. Uses SL4 method.
        """

        # Check if the mesh is periodic and a temporary node should be added
        if meshInfo.periodic and meshInfo.fix_periodic_domain_size and len(X) > 1:
            X = np.append(X, xf)  # Adding a temporary node for periodic mesh
            addedTempNode = True
        else:
            addedTempNode = False

        n = len(X)
        if n == 1:
            return LHS

        # Assume this function is defined elsewhere in Python
        centeredStencilLHS, centeredStencilRHS, decenteredStencilLHS, decenteredStencilRHS = self.__finiteDifferenceCoefficients(method)

        # Assume makeMatricesEachDirection is defined externally
        LHS_temp, RHS_temp = self.__makeMatricesEachDirection(
            centeredStencilLHS, centeredStencilRHS, 
            decenteredStencilLHS, decenteredStencilRHS, 
            [0], [n-1], n, None
        )

        # Solving for dXdEta
        #dXdEta = spsolve(LHS_temp[0], np.dot(RHS_temp[0], X))
        dXdEta = spsolve(LHS_temp[0], RHS_temp[0].dot(X))

        # Remove the temp node if it was added
        if addedTempNode:
            dXdEta = dXdEta[:-1]

        # Scale LHS matrix by dXdEta
        nTypes = len(LHS)
        for i in range(nTypes):
            #LHS[i] *= dXdEta[:, np.newaxis]  # Element-wise multiplication
            #LHS[i].data *= np.repeat(dXdEta, np.diff(LHS[i].indptr))
            LHS[i] = csr_matrix(LHS[i].multiply(dXdEta).tocsr())

        return LHS
    
    def __removeFilterEnds(self, numMethods, fLHS, fRHS, axis):
        startAttr = f'filter_borders_start_{axis.lower()}'
        endAttr = f'filter_borders_end_{axis.lower()}'

        #if hasattr(numMethods, startAttr) and not getattr(numMethods, startAttr):
        #    for i in range(len(fLHS)):
        #        fLHS[i][:5, :] = 0
        #        fRHS[i][:5, :] = 0
        #        fLHS[i][:5, :5] = np.eye(5)
        #        fRHS[i][:5, :5] = np.eye(5)

        #if hasattr(numMethods, endAttr) and not getattr(numMethods, endAttr):
        #    for i in range(len(fLHS)):
        #        fLHS[i][-5:, :] = 0
        #        fRHS[i][-5:, :] = 0
        #        fLHS[i][-5:, -5:] = np.eye(5)
        #        fRHS[i][-5:, -5:] = np.eye(5)


        if hasattr(numMethods, startAttr) and not getattr(numMethods, startAttr):
            for i in range(len(fLHS)):
                fLHS[i] = fLHS[i].tolil()
                fRHS[i] = fRHS[i].tolil()

                fLHS[i][0:5, :] = 0
                fRHS[i][0:5, :] = 0

                fLHS[i][0:5, 0:5] = np.eye(5)
                fRHS[i][0:5, 0:5] = np.eye(5)

                fLHS[i] = fLHS[i].tocsr()
                fRHS[i] = fRHS[i].tocsr()

        if hasattr(numMethods, endAttr) and not getattr(numMethods, endAttr):
            for i in range(len(fLHS)):
                fLHS[i] = fLHS[i].tolil()
                fRHS[i] = fRHS[i].tolil()

                n_rows, n_cols = fLHS[i].shape

                fLHS[i][n_rows - 5 : n_rows, :] = 0
                fRHS[i][n_rows - 5 : n_rows, :] = 0

                fLHS[i][n_rows - 5 : n_rows, n_cols - 5 : n_cols] = np.eye(5)
                fRHS[i][n_rows - 5 : n_rows, n_cols - 5 : n_cols] = np.eye(5)

                fLHS[i] = fLHS[i].tocsr()
                fRHS[i] = fRHS[i].tocsr()
        return fLHS, fRHS
    
    
    
    def getMatrixTypeBlocks(self, typeMap, p_row, p_col):
        # Define o tamanho máximo do bloco
        maxBlockSize = 128  # Máximo número de linhas por bloco, blocos maiores serão divididos

        # Inicializa a lista de blocos
        blocks = [[] for _ in range(p_row * p_col)]

        J, K = typeMap.shape

        # Obter todos os blocos
        allBlocks = []
        for k in range(K):
            starts = np.append([1], np.where(np.diff(typeMap[:, k]) != 0)[0] + 2)
            ends = np.append(starts[1:] - 1, J)
            allBlocks.append(np.column_stack((typeMap[starts-1, k] , starts, ends, np.full((len(starts), 2), k+1))))

        allBlocks = np.vstack(allBlocks)
        # Mesclar blocos
        i = 0
        while i < allBlocks.shape[0] - 1:
            j = i + 1
            while j < allBlocks.shape[0]:
                if np.all(allBlocks[i, 0:3] == allBlocks[j, 0:3]) and allBlocks[i, 4] + 1 == allBlocks[j, 3]:
                    allBlocks[i, 4] = allBlocks[j, 4]
                    allBlocks = np.delete(allBlocks, j, axis=0)
                else:
                    j += 1
            i += 1

        # Dividir blocos para processadores
        domainSlicesY = get_domain_slices(J, p_row)
        domainSlicesZ = get_domain_slices(K, p_col)

        for j in range(p_row):
            for k in range(p_col):
                nProc = k + j * p_col
                bL = allBlocks.copy()

                Ji = domainSlicesY[0][j]
                Jf = domainSlicesY[1][j]
                Ki = domainSlicesZ[0][k]
                Kf = domainSlicesZ[1][k]

                bL = [block for block in bL if not (block[1] > Jf or block[2] < Ji or block[3] > Kf or block[4] < Ki)]

                for block in bL:
                    block[1] = max(block[1], Ji)
                    block[2] = min(block[2], Jf)
                    block[3] = max(block[3], Ki)
                    block[4] = min(block[4], Kf)

                blocks[nProc] = bL

        # Reduzir tamanhos dos blocos, se necessário
        if maxBlockSize != float('inf'):
            for nProc in range(p_row * p_col):
                bL = blocks[nProc]
                bLnew = []
                for block in bL:
                    iSize = block[2] - block[1] + 1
                    jSize = block[4] - block[3] + 1
                    bSize = iSize * jSize
                    nSlices = int(np.ceil(bSize / maxBlockSize))
                    if nSlices == 1:
                        bLnew.append(block)
                    else:
                        iSlices = int(np.ceil(nSlices / jSize))
                        jSlices = int(np.floor(nSlices / iSlices))

                        iSlicesInd = get_domain_slices(iSize, iSlices)
                        jSlicesInd = get_domain_slices(jSize, jSlices)

                        iSlicesInd = (block[1] + iSlicesInd.T - 1).tolist()
                        jSlicesInd = (block[3] + jSlicesInd.T - 1).tolist()

                        for ii in range(iSlices):
                            for jj in range(jSlices):
                                bLnew.append([block[0], *iSlicesInd[ii], *jSlicesInd[jj]])

                blocks[nProc] = bLnew

        return blocks
    
def safe_flip(array):
    if isinstance(array, int):  # Verifica se o array é um valor inteiro
        return np.array([array])  # Converte o inteiro para um array 2D com um único valor
    elif isinstance(array, list):  # Verifica se o array é uma lista
        array = np.array(array)
    if array.size == 1:  # Verifica se o array tem apenas um elemento
        return array
    else:
        return np.flip(array, axis=(0, 1))  # Aplica o flip normalmente
    
def to_numpy_vector(value):
            if np.isscalar(value):
                return np.array([value])
            else:
                return np.array(value)
            
#TODO: Verificar a otimização disso
def sparse_operation_direct(eta, baseMatrixL, bufferMatrixL, i):
    # Obter a matriz esparsa diagonal (DIAgonal) de base e buffer
    baseMatrix_sparse = baseMatrixL[i]
    bufferMatrix_sparse = bufferMatrixL[i]
    
    num_diags = baseMatrix_sparse.data.shape[0]  # Número de diagonais
    
    # Criar uma nova matriz para armazenar as diagonais resultantes
    result_data = np.zeros_like(baseMatrix_sparse.data)
    
    # Loop sobre as diagonais
    for d in range(num_diags):
        base_diag = baseMatrix_sparse.data[d]  # Obter os valores da diagonal d
        buffer_diag = bufferMatrix_sparse.data[d]  # Obter os valores da diagonal d
        
        # Efetuar a operação diretamente sobre os valores da diagonal
        result_data[d] = eta[:, np.newaxis] * base_diag + (1 - eta[:, np.newaxis]) * buffer_diag
    
    # Criar uma nova matriz esparsa DIAgonal com as diagonais resultantes
    result_sparse = dia_matrix((result_data, baseMatrix_sparse.offsets), shape=baseMatrix_sparse.shape)
    
    return result_sparse

def ensure_iterable(var):
    if isinstance(var, int):
        return [var]
    return var

def get_array_dimensions(array):
            shape = np.shape(array)  # Obtém as dimensões do array
            if len(shape) == 0:  # Caso seja um número único
                return 1, 1
            elif len(shape) == 1:  # Caso seja um vetor de 1D
                return shape[0], 1
            else:
                return shape