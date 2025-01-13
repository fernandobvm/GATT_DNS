import os
import re
import numpy as np
from source.library import *
from source.BoundaryFiles import *
from source.MatricesFiles import *

class Preprocessing:
    def __init__(self, domain, mesh, flow_type, flow_parameters, numMethods, caseName, p_row, p_col, logAll, tridimensional, time, runningLST = False):
        self.domain = domain
        self.mesh = mesh
        self.flow_type = flow_type
        self.flow_parameters = flow_parameters
        self.numMethods = numMethods
        self.caseName = caseName
        self.p_row = p_row
        self.p_col = p_col
        self.logAll = logAll
        self.tridimensional = tridimensional
        self.time = time
        self.boundaryInfo = None
        self.matrices = None
        self.genInitialFlow = None
        self.SFD_X = None
        self.runningLST = runningLST

        self.run_preprocessing()

    def run_preprocessing(self):
        self.run_mesh()
        self.select_boundary_conditions()
        self.make_matrices()
        self.prepare_sfd()
        self.write_files()

    def run_mesh(self):
        # Add fixed points to mesh structure
        self.mesh.add_fixed_points(self.flow_type, self.domain)

        # Run mesh generator
        self.mesh.X, self.mesh.nx = self.mesh.generate_mesh(self.domain.xi, self.domain.xf, 'X')
        self.mesh.Y, self.mesh.ny = self.mesh.generate_mesh(self.domain.yi, self.domain.yf, 'Y')
        self.mesh.Z, self.mesh.nz = self.mesh.generate_mesh(self.domain.zi, self.domain.zf, 'Z')

    def select_boundary_conditions(self):
        self.boundary = BoundaryConditions(self.flow_type, self.mesh, self.flow_parameters, [self.numMethods.neumann_order, self.numMethods.neumann2_order])

        domainSlicesY = get_domain_slices(self.mesh.ny, self.p_row)
        domainSlicesZ = get_domain_slices(self.mesh.nz, self.p_col)

        self.boundaryInfo = init_boundaries(self.boundary, self.mesh, domainSlicesY, domainSlicesZ, self.p_row, self.p_col)

    def make_matrices(self):
        self.matrices = Matrices(self.mesh, self.domain, self.boundary, self.numMethods)

        # Prepare matrices for Thomas algorithm
        self.matrices.x = self.matrices.prepareThomas(self.matrices.x)
        self.matrices.x.blocks = self.matrices.getMatrixTypeBlocks(self.matrices.x.types, self.p_row, self.p_col)
        self.matrices.y = self.matrices.prepareThomas(self.matrices.y)
        self.matrices.y.blocks = self.matrices.getMatrixTypeBlocks(self.matrices.y.types, self.p_row, self.p_col)

        if self.mesh.nz > 1:
            self.matrices.z = self.matrices.prepareThomas(self.matrices.z)
            self.matrices.z.blocks = self.matrices.getMatrixTypeBlocks(self.matrices.z.types, self.p_row, self.p_col)

        self.matrices.neumann_coeffs = self.boundary.neumann_coeffs
        self.matrices.neumann2_coeffs = self.boundary.neumann2_coeffs

    def prepare_sfd(self):
        if hasattr(self.numMethods, 'SFD'):
            if self.numMethods.SFD.type == 2:
                self.SFD_X = self.calcSFDregion()

            if np.isinf(self.numMethods.SFD.Delta):
                self.numMethods.SFD.Delta = -1

            if (self.numMethods.SFD.type > 0) and (os.path.exists(f"{self.caseName}/meanflowSFD.h5" or self.flow_type.initial_meanFile is not None)):
                if self.numMethods.SFD.resume == None:
                    self.numMethods.SFD.resume = 1
            else:
                self.numMethods.SFD.resume = 0

    def write_files(self):
        if not os.path.exists(self.caseName):
            os.mkdir(self.caseName)

        # Save mesh to a file using numpy
        np.save(f"{self.caseName}/mesh.npy", {
            'X': self.mesh.X, 
            'Y': self.mesh.Y, 
            'Z': self.mesh.Z, 
            'wall': self.boundary.inside_wall, 
            'flow_parameters': self.flow_parameters,
            'flow_type': self.flow_type
        })

        # Check for previous save files, TODO: CHECK THIS GLOBALS
        if self.runningLST == False:
            nStep, nx, ny, nz = checkPreviousRun(self.caseName)

            if nStep is not None:
                self.genInitialFlow = False
                self.time.nStep = nStep

                if nx != self.mesh.nx or ny != self.mesh.ny or nz != self.mesh.nz:
                    raise ValueError(f"Mesh size has changed since last run: Parameters file indicates "
                                     f"{self.mesh.nx}x{self.mesh.ny}x{self.mesh.nz} but the file contains data for "
                                     f"{nx}x{ny}x{nz}")
            else:
                self.genInitialFlow = True
                self.time.nStep = 0
        else:
            self.genInitialFlow = False
            self.time.nStep = 0
            self.numMethods.SFD.resume = 0

        # Write Fortran files
        if not os.path.exists(f"{self.caseName}/bin"):
            os.mkdir(f"{self.caseName}/bin")

        disturbTypes = self.writeFortranDisturbances(self.caseName, self.boundaryInfo, self.tridimensional)
        self.writeFortranParameters(self.caseName, self.mesh, self.flow_parameters, self.time, self.numMethods, self.logAll, self.p_row, self.p_col)
        self.writeFortranMatrices(self.caseName, self.matrices, self.numMethods, self.mesh)
        self.writeFortranBoundaries(self.caseName, self.boundaryInfo)

        # SFD
        if self.numMethods.SFD.type == 2:
            # Save in format .npy
            np.save(f"{self.caseName}/bin/SFD_X.npy", {'SFD_X': self.SFD_X})
            # Save in format .hdf5
            with h5py.File(f"{self.caseName}/bin/SFD_X.h5", 'w') as hdf5_file:
                for key, value in {'SFD_X': self.SFD_X}.items():
                    hdf5_file.create_dataset(key, data=value.T)

    def calcSFDregion(self):
        # Inicializa o SFD_X com valores de 1
        SFD_X = np.ones((self.mesh.nx, self.mesh.ny, self.mesh.nz))

        # Define o valor padrão para applyX, applyY, applyZ
        if not hasattr(self.numMethods.SFD, 'applyX') or (isinstance(self.numMethods.SFD.applyX, bool) and self.numMethods.SFD.applyX):
            self.numMethods.SFD.applyX = 2
        if not hasattr(self.numMethods.SFD, 'applyY') or (isinstance(self.numMethods.SFD.applyY, bool) and self.numMethods.SFD.applyY):
            self.numMethods.SFD.applyY = 2
        if not hasattr(self.numMethods.SFD, 'applyZ') or (isinstance(self.numMethods.SFD.applyZ, bool) and self.numMethods.SFD.applyZ):
            self.numMethods.SFD.applyZ = 2

        # Aplicação de SFD nas direções X
        if self.numMethods.SFD.applyX == 2 or self.numMethods.SFD.applyX == -1:
            for i in range(self.mesh.x.buffer_i.n):
                SFD_X[i, :, :] *= (0.5 - 0.5 * np.cos(np.pi * i / (self.mesh.x.buffer_i.n)))
        
        if self.numMethods.SFD.applyX == 2 or self.numMethods.SFD.applyX == 1:
            for i in range(self.mesh.x.buffer_f.n):
                SFD_X[-(i + 1), :, :] *= (0.5 - 0.5 * np.cos(np.pi * i / (self.mesh.x.buffer_f.n)))

        # Aplicação de SFD nas direções Y
        if self.numMethods.SFD.applyY == 2 or self.numMethods.SFD.applyY == -1:
            for j in range(self.mesh.y.buffer_i.n):
                SFD_X[:, j, :] *= (0.5 - 0.5 * np.cos(np.pi * j / (self.mesh.y.buffer_i.n)))

        if self.numMethods.SFD.applyY == 2 or self.numMethods.SFD.applyY == 1:
            for j in range(self.mesh.y.buffer_f.n):
                SFD_X[:, -(j + 1), :] *= (0.5 - 0.5 * np.cos(np.pi * j / (self.mesh.y.buffer_f.n)))

        # Aplicação de SFD nas direções Z
        if self.numMethods.SFD.applyZ == 2 or self.numMethods.SFD.applyZ == -1:
            for k in range(self.mesh.z.buffer_i.n):
                SFD_X[:, :, k] *= (0.5 - 0.5 * np.cos(np.pi * k / (self.mesh.z.buffer_i.n)))

        if self.numMethods.SFD.applyZ == 2 or self.numMethods.SFD.applyZ == 1:
            for k in range(self.mesh.z.buffer_f.n):
                SFD_X[:, :, -(k + 1)] *= (0.5 - 0.5 * np.cos(np.pi * k / (self.mesh.z.buffer_f.n)))

        # Aplica o fator multiplicativo final no SFD_X
        SFD_X = self.numMethods.SFD.X * (1 - SFD_X)

        # Se houver uma extraRegion, aplica os ajustes adicionais
        if hasattr(self.numMethods.SFD, 'extraRegion'):
            for extra_region in self.numMethods.SFD.extraRegion:
                ER = extra_region
                # Calcula o raio a partir do ponto central
                R = np.sqrt((self.mesh.X - ER.location[0])**2 / ER.size[0]**2 +
                            (self.mesh.Y - ER.location[1])**2 / ER.size[1]**2 +
                            (np.transpose(self.mesh.Z, (0, 2, 1)) - ER.location[2])**2 / ER.size[2]**2)
                
                R[R > 1] = 1
                R = 0.5 + 0.5 * np.cos(np.pi * R)
                SFD_X += ER.X * R
        
        return SFD_X

    
    def writeFortranDisturbances(self, caseName, bi, tridimensional):
        """
        Esta função cria os arquivos runDisturbances.F90 e disturbances.F90
        """
        # Número de processos
        nProcs = len(bi)

        # Criação dos arquivos runDisturbances.F90 e runForcings.F90
        outFileDisturb = open(f'{caseName}/bin/runDisturbances.F90', 'w')
        if not tridimensional:
            outFileForcing = open(f'{caseName}/bin/runForcings2D.F90', 'w')
            open(f'{caseName}/bin/runForcings3D.F90', 'a').close()  # Cria o arquivo vazio
        else:
            outFileForcing = open(f'{caseName}/bin/runForcings3D.F90', 'w')
            open(f'{caseName}/bin/runForcings2D.F90', 'a').close()  # Cria o arquivo vazio

        outFileDisturb.write('    select case (nrank)\n')
        outFileForcing.write('    select case (nrank)\n')

        disturbTypes = []

        # Loop pelos processos
        for i in range(nProcs):
            outFileDisturb.write(f'        case ({i})\n')
            outFileForcing.write(f'        case ({i})\n')

            for j in range(len(bi[i].disturb)):
                di = bi[i].disturb[j]

                # Decide se o arquivo é de forcing ou distúrbio
                outFile = outFileForcing if di.forcing else outFileDisturb

                disturbTypes.append(di.type)

                nx = di.ind[1] - di.ind[0] + 1
                ny = di.ind[3] - di.ind[2] + 1
                nz = di.ind[5] - di.ind[4] + 1

                outFile.write(f'            call {di.type}({nx},{ny},{nz},(/')

                # Escreve os valores de X
                for k in range(nx - 1):
                    outFile.write(f'{di.X[k]:.20f}d0,')
                outFile.write(f'{di.X[-1]:.20f}d0/),(/')

                # Escreve os valores de Y
                for k in range(ny - 1):
                    outFile.write(f'{di.Y[k]:.20f}d0,')
                outFile.write(f'{di.Y[-1]:.20f}d0/),(/')

                # Escreve os valores de Z
                for k in range(nz - 1):
                    outFile.write(f'{di.Z[k]:.20f}d0,')
                if di.forcing:
                    outFile.write(f'{di.Z[-1]:.20f}d0/)')
                else:
                    outFile.write(f'{di.Z[-1]:.20f}d0/),t')
                # Escreve as variáveis (vars)
                for var in di.var:
                    outFile.write(f',{var}({di.ind[0]}:{di.ind[1]},{di.ind[2]}:{di.ind[3]},{di.ind[4]}:{di.ind[5]})')

                if di.forcing:
                    for var in di.var:
                        outFile.write(f',d{var}({di.ind[0]}:{di.ind[1]},{di.ind[2]}:{di.ind[3]},{di.ind[4]}:{di.ind[5]})')

                # Escreve os parâmetros adicionais (par)
                for par in di.par:
                    if isinstance(par, (int, float)):  # Se for número
                        outFile.write(f',{par:.20f}d0')
                    elif isinstance(par, list):  # Se for vetor
                        outFile.write(',(/')
                        for k in range(len(par) - 1):
                            outFile.write(f'{par[k]:.20f}d0,')
                        outFile.write(f'{par[-1]:.20f}d0/)')
                    else:  # Se for string
                        outFile.write(f',\'{par}\'')

                outFile.write(')\n')

        outFileForcing.write('    end select\n')
        outFileDisturb.write('    end select\n')

        outFileForcing.close()
        outFileDisturb.close()

        # Remover duplicados em disturbTypes
        disturbTypes = list(set(disturbTypes))

        # Criação do arquivo disturbances.F90
        outFile = open(f'{caseName}/bin/disturbances.F90', 'w')
        outFile.write('    module disturbances\n\n    contains\n\n')

        for disturbType in disturbTypes:
            sourcePath = f'source/disturbances/{disturbType}.F90' if disturbType != 'holdInlet' else f'source/Fortran/{disturbType}.F90'
            with open(sourcePath, 'r') as sourceFile:
                for line in sourceFile:
                    outFile.write(line)
                outFile.write('\n')

        outFile.write('\n    end module\n')
        outFile.close()

        return disturbTypes

    def writeFortranParameters(self, caseName, mesh, flowParameters, time, numMethods, logAll, p_row, p_col):
        """
        This method writes the parameters.F90 file, which contains basic parameters that will be used in the simulation.
        """

        # Open the output file in write mode
        with open(f'{caseName}/bin/parameters.F90', 'w') as outFile:
            # Mesh parameters
            print('-------')
            outFile.write(f'    integer :: nx = {mesh.nx}\n')
            outFile.write(f'    integer :: ny = {mesh.ny}\n')
            outFile.write(f'    integer :: nz = {mesh.nz}\n\n')

            # Flow parameters
            outFile.write(f'    real*8 :: Re = {flowParameters.Re:.20f}d0\n')
            outFile.write(f'    real*8 :: Ma = {flowParameters.Ma:.20f}d0\n')
            outFile.write(f'    real*8 :: Pr = {flowParameters.Pr:.20f}d0\n')
            outFile.write(f'    real*8 :: T0 = {flowParameters.T0:.20f}d0\n')
            outFile.write(f'    real*8 :: gamma = {flowParameters.gamma:.20f}d0\n\n')

            # Time parameters
            outFile.write(f'    real*8 :: dtmax = {time.dt:.20f}d0\n')
            outFile.write(f'    real*8 :: maxCFL = {time.max_cfl:.20f}d0\n')

            if logAll == 0:
                logAll = 2147483647  # Maximum value for 32-bit integer

            outFile.write(f'    integer :: logAll = {logAll}\n')
            outFile.write(f'    integer :: nSave = {time.nStep}\n')

            # tracked_norm
            if not hasattr(mesh, 'tracked_norm') or not mesh.tracked_norm:
                outFile.write('    real*8 :: trackedNorm = 0.d0\n')
            else:
                trackedNorm = 1 / ((flowParameters.gamma**2 - flowParameters.gamma) * flowParameters.Ma**2)
                outFile.write(f'    real*8 :: trackedNorm = {trackedNorm:.20f}d0\n')

            # Time control
            if time.control == 'dt':
                outFile.write('    integer :: timeControl = 1\n')
                outFile.write(f'    integer :: qTimesInt = {time.qtimes}\n')
                outFile.write('    real*8  :: qTimesReal\n\n')
                outFile.write(f'    integer :: tmaxInt = {time.tmax}\n')
                outFile.write('    real*8  :: tmaxReal\n\n')
            elif time.control == 'cfl':
                outFile.write('    integer :: timeControl = 2\n')
                outFile.write('    integer :: qTimesInt\n')
                outFile.write(f'    real*8  :: qtimesReal = {time.qtimes:.20f}d0\n\n')
                outFile.write('    integer :: tmaxInt\n')
                outFile.write(f'    real*8  :: tmaxReal = {time.tmax:.20f}d0\n\n')
            else:
                raise ValueError('Unrecognized type of time control. Use either "dt" or "cfl".')

            # Time stepping method
            if numMethods.time_stepping == 'RK4':
                outFile.write('    integer :: timeStepping = 1\n')
            elif numMethods.time_stepping == 'Euler':
                outFile.write('    integer :: timeStepping = 2\n')
            elif numMethods.time_stepping == 'SSPRK3':
                outFile.write('    integer :: timeStepping = 3\n')
            else:
                raise ValueError('Unrecognized time stepping method.')

            # SFD method
            if hasattr(numMethods, 'SFD'):
                outFile.write(f'    integer :: SFD = {numMethods.SFD.type}\n')
                outFile.write(f'    real*8 :: SFD_Delta = {numMethods.SFD.Delta:.20f}d0\n')
                outFile.write(f'    real*8 :: SFD_X_val = {numMethods.SFD.X:.20f}d0\n')
                outFile.write(f'    integer :: resumeMeanFlow = {numMethods.SFD.resume:}\n\n')
            else:
                outFile.write('    integer :: SFD = 0\n')
                outFile.write(f'    real*8 :: SFD_Delta = {0:.20f}d0\n')
                outFile.write(f'    real*8 :: SFD_X_val = {0:.20f}d0\n')
                outFile.write('    integer :: resumeMeanFlow = 0\n\n')

            # Filter characteristic time
            if not hasattr(numMethods, 'spatial_filter_time') or numMethods.spatial_filter_time <= 0:
                outFile.write(f'    real*8 :: FilterCharTime = {-1:.20f}d0\n')
            else:
                outFile.write(f'    real*8 :: FilterCharTime = {numMethods.spatial_filter_time:.20f}d0\n')

            # Calculate dxmin
            dxmin = [
                1 / np.min(np.diff(mesh.X)),
                1 / np.min(np.diff(mesh.Y)),
                0 if mesh.nz == 1 else 1 / np.min(np.diff(mesh.Z))
            ]

            if hasattr(time, 'CFLignoreZ') and time.CFLignoreZ:
                dxmin[2] = 0

            outFile.write(f'    real*8,dimension(3) :: dxmin = (/{dxmin[0]:.20f}d0,{dxmin[1]:.20f}d0,{dxmin[2]:.20f}d0/)\n\n')

            # Process row and col
            outFile.write(f'    integer :: p_row = {p_row}\n')
            outFile.write(f'    integer :: p_col = {p_col}\n')

        # File is closed automatically after exiting the 'with' block

    def writeFortranMatrices(self, case_name, matrices, num_methods, mesh):
        # Abre o arquivo para escrita
        out_file = open(f'{case_name}/bin/matrices.F90', 'w')

        I = matrices.y.types.shape[0]
        J = matrices.x.types.shape[0]
        K = matrices.x.types.shape[1]
        n_procs = len(matrices.x.blocks)

        # Escrever blocos
        out_file.write('    select case(nrank)\n')

        for i in range(n_procs):
            out_file.write(f'        case({i})\n')

            # Para X
            blocks = np.array(matrices.x.blocks[i])
            n_blocks = blocks.shape[0]
            out_file.write(f'            nDerivBlocksX = {n_blocks}\n')
            out_file.write(f'            allocate(derivBlocksX({n_blocks},5))\n')

            out_file.write('            derivBlocksX = reshape((/')
            blocks = np.reshape(blocks.transpose(), (1, 5*n_blocks))
            for n in range(blocks.shape[1]-1):
                out_file.write(f'{blocks[0,n]},')
            out_file.write(f'{blocks[0,-1]}/),shape(derivBlocksX))\n\n')
            

            # Para Y
            blocks = np.array(matrices.y.blocks[i])
            n_blocks = blocks.shape[0]
            out_file.write(f'            nDerivBlocksY = {n_blocks}\n')
            out_file.write(f'            allocate(derivBlocksY({n_blocks},5))\n')

            out_file.write('            derivBlocksY = reshape((/')
            blocks = np.reshape(blocks.transpose(), (1, 5*n_blocks))
            for n in range(blocks.shape[1]-1):
                out_file.write(f'{blocks[0,n]},') 
            out_file.write(f'{blocks[0,-1] }/),shape(derivBlocksY))\n\n')

            # Para Z, se K > 1
            if K > 1:
                blocks = np.array(matrices.z.blocks[i])
                n_blocks = blocks.shape[0]
                out_file.write(f'            nDerivBlocksZ = {n_blocks}\n')
                out_file.write(f'            allocate(derivBlocksZ({n_blocks},5))\n')

                out_file.write('            derivBlocksZ = reshape((/')
                blocks = np.reshape(blocks.transpose(), (1, 5*n_blocks))
                for n in range(blocks.shape[1]-1):
                    out_file.write(f'{blocks[0,n]},') 
                out_file.write(f'{blocks[0,-1]}/),shape(derivBlocksZ))\n\n')

        out_file.write('    end select\n\n')

        # Escrever informações do filtro
        out_file.write(f'    filterX = {num_methods.filter_directions[0]}\n')
        out_file.write(f'    filterY = {num_methods.filter_directions[1]}\n')
        out_file.write(f'    filterZ = {num_methods.filter_directions[2]}\n')

        # Escrever matrizes para derivadas em X
        out_file.write(f'    derivnRHSx = {matrices.x.nRHS}\n')
        out_file.write(f'    filternRHSx = {matrices.x.nRHSf}\n')

        out_file.write(f'    allocate(derivsAX({I-1},{matrices.x.nTypes}))\n')
        out_file.write(f'    allocate(derivsBX({I-matrices.x.periodic},{matrices.x.nTypes}))\n')
        out_file.write(f'    allocate(derivsCX({I-1},{matrices.x.nTypes}))\n')
        out_file.write(f'    allocate(derivsRX({I},{2*matrices.x.nRHS-1},{matrices.x.nTypes}))\n\n')

        out_file.write(f'    allocate(filterAX({I-1},{matrices.x.nTypes}))\n')
        out_file.write(f'    allocate(filterBX({I-matrices.x.periodic},{matrices.x.nTypes}))\n')
        out_file.write(f'    allocate(filterCX({I-1},{matrices.x.nTypes}))\n')
        out_file.write(f'    allocate(filterRX({I},{2*matrices.x.nRHSf-1},{matrices.x.nTypes}))\n\n')

        # Escrever as matrizes
        self.write_matrix(out_file, 'derivsAX', matrices.x.A)
        self.write_matrix(out_file, 'derivsBX', matrices.x.B)
        self.write_matrix(out_file, 'derivsCX', matrices.x.C)
        self.write_matrix(out_file, 'derivsRX', matrices.x.R)

        self.write_matrix(out_file, 'filterAX', matrices.x.Af)
        self.write_matrix(out_file, 'filterBX', matrices.x.Bf)
        self.write_matrix(out_file, 'filterCX', matrices.x.Cf)
        self.write_matrix(out_file, 'filterRX', matrices.x.Rf)

        out_file.write(f'    periodicX = {int(matrices.x.periodic)}\n\n')

        if matrices.x.periodic:
            out_file.write(f'    allocate(derivsDX({I},{matrices.x.nTypes}))\n')
            out_file.write(f'    allocate(filterDX({I},{matrices.x.nTypes}))\n')

            self.write_matrix(out_file, 'derivsDX', matrices.x.D)
            self.write_matrix(out_file, 'filterDX', matrices.x.Df)

        # Derivadas em Y
        out_file.write(f'    derivnRHSy = {matrices.y.nRHS}\n')
        out_file.write(f'    filternRHSy = {matrices.y.nRHSf}\n')

        out_file.write(f'    allocate(derivsAY({J-1},{matrices.y.nTypes}))\n')
        out_file.write(f'    allocate(derivsBY({J-matrices.y.periodic},{matrices.y.nTypes}))\n')
        out_file.write(f'    allocate(derivsCY({J-1},{matrices.y.nTypes}))\n')
        out_file.write(f'    allocate(derivsRY({J},{2*matrices.y.nRHS-1},{matrices.y.nTypes}))\n\n')

        out_file.write(f'    allocate(filterAY({J-1},{matrices.y.nTypes}))\n')
        out_file.write(f'    allocate(filterBY({J-matrices.y.periodic},{matrices.y.nTypes}))\n')
        out_file.write(f'    allocate(filterCY({J-1},{matrices.y.nTypes}))\n')
        out_file.write(f'    allocate(filterRY({J},{2*matrices.y.nRHSf-1},{matrices.y.nTypes}))\n\n')

        self.write_matrix(out_file, 'derivsAY', matrices.y.A)
        self.write_matrix(out_file, 'derivsBY', matrices.y.B)
        self.write_matrix(out_file, 'derivsCY', matrices.y.C)
        self.write_matrix(out_file, 'derivsRY', matrices.y.R)

        self.write_matrix(out_file, 'filterAY', matrices.y.Af)
        self.write_matrix(out_file, 'filterBY', matrices.y.Bf)
        self.write_matrix(out_file, 'filterCY', matrices.y.Cf)
        self.write_matrix(out_file, 'filterRY', matrices.y.Rf)

        out_file.write(f'    periodicY = {int(matrices.y.periodic)}\n\n')

        if matrices.y.periodic:
            out_file.write(f'    allocate(derivsDY({J},{matrices.y.nTypes}))\n')
            out_file.write(f'    allocate(filterDY({J},{matrices.y.nTypes}))\n')

            self.write_matrix(out_file, 'derivsDY', matrices.y.D)
            self.write_matrix(out_file, 'filterDY', matrices.y.Df)

        # Derivadas em Z
        if K > 1:
            out_file.write(f'    derivnRHSz = {matrices.z.nRHS}\n')
            out_file.write(f'    filternRHSz = {matrices.z.nRHSf}\n')

            out_file.write(f'    allocate(derivsAZ({K-1},{matrices.z.nTypes}))\n')
            out_file.write(f'    allocate(derivsBZ({K-matrices.z.periodic},{matrices.z.nTypes}))\n')
            out_file.write(f'    allocate(derivsCZ({K-1},{matrices.z.nTypes}))\n')
            out_file.write(f'    allocate(derivsRZ({K},{2*matrices.z.nRHS-1},{matrices.z.nTypes}))\n\n')

            out_file.write(f'    allocate(filterAZ({K-1},{matrices.z.nTypes}))\n')
            out_file.write(f'    allocate(filterBZ({K-matrices.z.periodic},{matrices.z.nTypes}))\n')
            out_file.write(f'    allocate(filterCZ({K-1},{matrices.z.nTypes}))\n')
            out_file.write(f'    allocate(filterRZ({K},{2*matrices.z.nRHSf-1},{matrices.z.nTypes}))\n\n')

            self.write_matrix(out_file, 'derivsAZ', matrices.z.A)
            self.write_matrix(out_file, 'derivsBZ', matrices.z.B)
            self.write_matrix(out_file, 'derivsCZ', matrices.z.C)
            self.write_matrix(out_file, 'derivsRZ', matrices.z.R)

            self.write_matrix(out_file, 'filterAZ', matrices.z.Af)
            self.write_matrix(out_file, 'filterBZ', matrices.z.Bf)
            self.write_matrix(out_file, 'filterCZ', matrices.z.Cf)
            self.write_matrix(out_file, 'filterRZ', matrices.z.Rf)

            out_file.write(f'    periodicZ = {int(matrices.z.periodic)}\n\n')

            if matrices.z.periodic:
                out_file.write(f'    allocate(derivsDZ({K},{matrices.z.nTypes}))\n')
                out_file.write(f'    allocate(filterDZ({K},{matrices.z.nTypes}))\n')

                self.write_matrix(out_file, 'derivsDZ', matrices.z.D)
                self.write_matrix(out_file, 'filterDZ', matrices.z.Df)

        # Neumann Coefficients
        neumann_coeffs = matrices.neumann_coeffs  
        out_file.write(f"\n    neumannLength = {len(neumann_coeffs)}\n")
        out_file.write(f"    allocate(neumannCoeffs({len(neumann_coeffs)}))\n")
        out_file.write("    neumannCoeffs = (/")
        out_file.write(",".join([f"{coeff:.20f}d0" for coeff in neumann_coeffs[:-1]]))
        out_file.write(f",{neumann_coeffs[-1]:.20f}d0/)\n\n")

        # Neumann2 Coefficients
        neumann2_coeffs = matrices.neumann2_coeffs  # Atribua a matriz correspondente aqui
        out_file.write(f"\n    neumann2Length = {len(neumann2_coeffs)}\n")
        out_file.write(f"    allocate(neumann2Coeffs({len(neumann2_coeffs)}))\n")
        out_file.write("    neumann2Coeffs = (/")
        out_file.write(",".join([f"{coeff:.20f}d0" for coeff in neumann2_coeffs[:-1]]))
        out_file.write(f",{neumann2_coeffs[-1]:.20f}d0/)\n")

        # Tracked Points
        if not hasattr(mesh, 'tracked_points') or len(mesh.tracked_points) == 0:
            out_file.write("    nTracked = 0\n")
        else:
            tracked_points = mesh.tracked_points  # Atribua os pontos rastreados
            n_tracked = tracked_points.shape[0]
            out_file.write(f"    nTracked = {n_tracked}\n")
            out_file.write(f"    allocate(indTracked({n_tracked},3))\n")
            
            # Handling mesh refinement
            x_temp = np.full_like(mesh.X, np.nan)
            y_temp = np.full_like(mesh.Y, np.nan)
            z_temp = np.full_like(mesh.Z, np.nan)
            
            step_x = mesh.x.extra_refinement + 1
            step_y = mesh.y.extra_refinement + 1
            step_z = mesh.z.extra_refinement + 1
            
            x_temp[::step_x] = mesh.X[::step_x]
            y_temp[::step_y] = mesh.Y[::step_y]
            z_temp[::step_z] = mesh.Z[::step_z]
            
            ind_tracked = np.empty_like(tracked_points, dtype=float)
            
            for i in range(n_tracked):
                ind_tracked[i, 0] = np.argmin(np.abs(tracked_points[i, 0] - x_temp))
                ind_tracked[i, 1] = np.argmin(np.abs(tracked_points[i, 1] - y_temp))
                ind_tracked[i, 2] = np.argmin(np.abs(tracked_points[i, 2] - z_temp))
            
            # Write reshaped indices to the file
            ind_tracked_flat = ind_tracked.astype(int).T.flatten()
            out_file.write("    indTracked = reshape((/")
            out_file.write(",".join(map(str, ind_tracked_flat[:-1] + 1)))
            out_file.write(f",{ind_tracked_flat[-1] + 1}/),shape(indTracked))\n")

        # Fecha o arquivo
        #out_file.write('    return\nend subroutine prepareMatrices\n')
        out_file.close()

    def write_matrix(self, out_file, var_name, matrix):
        # Função auxiliar para escrever uma matriz no arquivo
        flattened = matrix.T.flatten()
        out_file.write(f'    {var_name} = reshape((/')
        for n in range(flattened.shape[0] - 1):
            out_file.write(f'{flattened[n]:.20f}d0,')
        out_file.write(f'{flattened[-1]}d0/),shape({var_name}))\n')

    def writeFortranBoundaries(self, caseName, bi):
        """
        This method writes the boundaryInfo file with data for all boundary conditions for each domain slice.
        The file is saved in the caseName/bin/ directory in Fortran-compatible format.
        """
        nProcs = len(bi)

        # Create the output directory if it doesn't exist
        output_dir = os.path.join(caseName, 'bin')
        os.makedirs(output_dir, exist_ok=True)

        # Open the output file for writing
        with open(os.path.join(output_dir, 'boundaryInfo.F90'), 'w') as outFile:
            outFile.write('    select case (nrank)\n')

            vars = ['U', 'V', 'W', 'E', 'P']

            for i in range(nProcs):

                outFile.write(f'        case ({i})\n')

                # Dirichlet Boundary Conditions
                for j, var in enumerate(vars):
                    if j == 0:
                        n, ind, val = bi[i].nUd, bi[i].iUd, bi[i].vUd
                    elif j == 1:
                        n, ind, val = bi[i].nVd, bi[i].iVd, bi[i].vVd
                    elif j == 2:
                        n, ind, val = bi[i].nWd, bi[i].iWd, bi[i].vWd
                    elif j == 3:
                        n, ind, val = bi[i].nEd, bi[i].iEd, bi[i].vEd
                    elif j == 4:
                        n, ind, val = bi[i].nPd, bi[i].iPd, bi[i].vPd

                    outFile.write(f'            n{var}d = {n}\n')

                    if n > 0:
                        outFile.write(f'            allocate(i{var}d({n},6))\n')
                        outFile.write(f'            allocate(v{var}d({n}))\n')

                        outFile.write(f'            i{var}d = reshape((/')
                        outFile.write(','.join([f'{i}' for i in ind.T.flatten()[:-1]]))
                        outFile.write(f',{ind.T.flatten()[-1]}/),shape(i{var}d))\n')

                        outFile.write(f'            v{var}d = (/')
                        if len(val) > 1:
                            outFile.write(','.join([f'{v:.20f}d0' for v in val[:-1]]))
                            outFile.write(f',{val[-1]:.20f}d0/)\n\n')
                        else:
                            outFile.write(f'{val[-1]:.20f}d0/)\n\n')

                # Neumann Boundary Conditions
                for j, var in enumerate(vars):
                    if j == 0:
                        n, ind, dir_ = bi[i].nUn, bi[i].iUn, bi[i].dUn
                    elif j == 1:
                        n, ind, dir_ = bi[i].nVn, bi[i].iVn, bi[i].dVn
                    elif j == 2:
                        n, ind, dir_ = bi[i].nWn, bi[i].iWn, bi[i].dWn
                    elif j == 3:
                        n, ind, dir_ = bi[i].nEn, bi[i].iEn, bi[i].dEn
                    elif j == 4:
                        n, ind, dir_ = bi[i].nPn, bi[i].iPn, bi[i].dPn

                    outFile.write(f'            n{var}n = {n}\n')

                    if n > 0:
                        outFile.write(f'            allocate(i{var}n({n},6))\n')
                        outFile.write(f'            allocate(d{var}n({n}))\n')

                        outFile.write(f'            i{var}n = reshape((/')
                        outFile.write(','.join([f'{i}' for i in ind.T.flatten()[:-1]]))
                        outFile.write(f',{ind.T.flatten()[-1]}/),shape(i{var}n))\n')

                        outFile.write(f'            d{var}n = (/')
                        if len(dir_) > 1:
                            outFile.write(','.join(map(str, dir_[:-1])))
                            outFile.write(f',{dir_[-1]}/)\n\n')
                        else:
                            outFile.write(f'{dir_[-1]}/)\n\n')

                    else:
                        outFile.write(f'            allocate(i{var}n(1,6))\n')
                        outFile.write(f'            allocate(d{var}n(1))\n\n')

                # Second Derivative Boundary Conditions
                for j, var in enumerate(vars):
                    if j == 0:
                        n, ind, dir_ = bi[i].nUs, bi[i].iUs, bi[i].dUs
                    elif j == 1:
                        n, ind, dir_ = bi[i].nVs, bi[i].iVs, bi[i].dVs
                    elif j == 2:
                        n, ind, dir_ = bi[i].nWs, bi[i].iWs, bi[i].dWs
                    elif j == 3:
                        n, ind, dir_ = bi[i].nEs, bi[i].iEs, bi[i].dEs
                    elif j == 4:
                        n, ind, dir_ = bi[i].nPs, bi[i].iPs, bi[i].dPs

                    outFile.write(f'            n{var}s = {n}\n')

                    if n > 0:
                        outFile.write(f'            allocate(i{var}s({n},6))\n')
                        outFile.write(f'            allocate(d{var}s({n}))\n')

                        outFile.write(f'            i{var}s = reshape((/')
                        outFile.write(','.join([f'{i}' for i in ind.T.flatten()[:-1]]))
                        outFile.write(f',{ind.T.flatten()[-1]}/),shape(i{var}s))\n')

                        outFile.write(f'            d{var}s = (/')
                        if len(dir_)>1:
                            outFile.write(','.join(map(str, dir_[:-1])))
                            outFile.write(f',{dir_[-1]}/)\n\n')
                        else:
                            outFile.write(f'{dir_[-1]}/)\n\n')

                    else:
                        outFile.write(f'            allocate(i{var}s(1,6))\n')
                        outFile.write(f'            allocate(d{var}s(1))\n\n')

                # Corners
                outFile.write(f'            cN = {bi[i].cN}\n')
                outFile.write(f'            allocate(cL({max(bi[i].cN,1)},6))\n')
                outFile.write(f'            allocate(cD({max(bi[i].cN,1)},3))\n')
                outFile.write(f'            allocate(cAdiabatic({max(bi[i].cN,1)}))\n')

                if bi[i].cN > 0:
                    outFile.write(f'            cL = reshape((/')
                    outFile.write(','.join([f'{k:.0f}' for k in bi[i].cL.T.flatten()[:-1]]))
                    outFile.write(f',{bi[i].cL.T.flatten()[-1]:.0f}/),shape(cL))\n')

                    outFile.write(f'            cD = reshape((/')
                    outFile.write(','.join([f'{k:.0f}' for k in bi[i].cD.T.flatten()[:-1]]))
                    outFile.write(f',{bi[i].cD.T.flatten()[-1]:.0f}/),shape(cD))\n')

                    outFile.write(f'            cAdiabatic = (/')
                    outFile.write(','.join([f'{k:.0f}' for k in bi[i].adiabatic.T.flatten()[:-1]]))
                    outFile.write(f',{bi[i].adiabatic.T.flatten()[-1]:.0f}/)\n\n')

            outFile.write('    end select\n')

        # Save the boundary conditions in a .npy format for future reference
        np.save(os.path.join(output_dir, 'boundary_conditions.npy'), bi)



