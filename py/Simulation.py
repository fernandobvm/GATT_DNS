import os
import shutil
import subprocess
import time
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from datetime import datetime
from scipy import io
from source.library import *
from source.MatricesFiles import *
from source.BoundaryFiles import *
from source.preprocessing import *

def runDNS(caseName, flowParameters, domain, flowType, mesh, time, numMethods, p_row, p_col, caseFile, script_path, logAll = False, runSimulation = True, compileCode = True, plotDNSDomain = False, forceRecompileAll = False, displayCompiling = False, optimizeCode = True, debugger = False, profiler = False, matlabDir = '', decompDir = '/usr/local/2decomp_fft', extraParameters = None):

    if extraParameters is not None:
        extraVars = unpackStruct(extraParameters)
        for var in extraVars:
            exec(f"{var} = extraParameters['{var}']")

    # Verify if is 2D or 3D
    tridimensional = mesh.z.n + mesh.z.buffer_i.n + mesh.z.buffer_f.n > 1
    if not tridimensional:
        p_col = 1

    # Set paths to files
    source_path = 'source'
    case_path = f'../runs/{caseName}'
    boundaries_path = 'source/boundaries'
    
    if source_path not in os.sys.path:
        os.sys.path.append(source_path)
    if boundaries_path not in os.sys.path:
        os.sys.path.append(boundaries_path)

    if not os.path.exists(case_path):
        os.mkdir(case_path)

    print(f'Parameters file: {caseFile}\nCase name: {caseName}')

    # Intialize log file
    log_file_path = os.path.join(case_path, 'log.txt')
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as logFile:
            logFile.write('Save number\tIteration\tSimulation time\tdt       \tCFL      \tU change\tV change\tW change\tR change\tE change')
            if hasattr(mesh, 'trackedPoints'):
                for i in range(mesh.trackedPoints.shape[0]):
                    logFile.write(f'\tU{i+1}\tV{i+1}\tW{i+1}\tR{i+1}\tE{i+1}')
            logFile.write('\n')

    bin_path = f'{case_path}/bin'
    if not os.path.exists(bin_path):
        os.mkdir(bin_path)

    # Copy parameters file to bin folder
    os.system(f'cp {script_path} {bin_path}')
    # Copy phi.dat for white noise simulations
    os.system(f'cp ../source/disturbances/phi.dat {bin_path}')

    if forceRecompileAll:
        compileCode = True
        for ext in ['.mod', '.o']:
            for file in os.listdir(bin_path):
                if file.endswith(ext):
                    os.remove(os.path.join(bin_path, file))

    print('Running preprocessor')
    p = Preprocessing(domain, mesh, flowType, flowParameters, numMethods, case_path, p_row, p_col, logAll, tridimensional, time)
    print(p.time.nStep)
    print(f'Mesh size: {mesh.nx} x {mesh.ny} x {mesh.nz}')

    # Generates initial flow, if needed
    if p.genInitialFlow: 
        print('Generating new initial flow')

        flow = generateInitialFlow(mesh, flowParameters, flowType, p.boundary.inside_wall, flowType.name)

        flow_to_save = flow
        flow_to_save.t = 0

        for var in ['U', 'V', 'W', 'R', 'E']:
            current_array = getattr(flow_to_save, var)
            if current_array.size > 0: 
                current_array[p.boundary.inside_wall.T] = np.nan
                setattr(flow_to_save, var, current_array) 

        # Saving initial flow in .npy and .h5
        #np.save(f'{case_path}/flow_0000000000.npy', flow_to_save)

        # Save using h5
        with h5py.File(f'{case_path}/flow_0000000000.h5', 'w') as hdf5_file:
            for attr in ['U', 'V', 'W', 'R', 'E', 't']:
                data = getattr(flow_to_save, attr)
                if isinstance(data, np.ndarray): 
                    data = np.squeeze(data).T
                if attr != 't':
                    hdf5_file.create_dataset(attr, data=data, compression="gzip", compression_opts=9) #compression opts goes from 1 to 9 (1 is the less compressed)
                else:
                    hdf5_file.create_dataset(attr, data=data)

        if flowType.initial_meanFile != None:
            flowTypeTemp = FlowType()  
            flowTypeTemp.initial_type = 'file'
            flowTypeTemp.initial_flowFile = flowType.initial_meanFile
            if flowType.initial_meshFile != None:
                flowTypeTemp.initial_meshFile = flowType.initial_meshFile

            meanFlow = generateInitialFlow(mesh, flowParameters, flowTypeTemp, p.boundary.inside_wall, flowType.name)

            meanFlow_to_save = deepcopy(meanFlow)
            meanFlow_to_save.t = 0

            for var in ['U', 'V', 'W', 'R', 'E']:
                current_array = getattr(meanFlow_to_save, var)
                if current_array.size > 0:
                    current_array[p.boundary.inside_wall.T] = np.nan
                    setattr(meanFlow_to_save, var, current_array)


            #np.save(f'{case_path}/meanflowSFD.npy', meanFlow_to_save)
            with h5py.File(f'{case_path}/meanflowSFD.h5', 'w') as hdf5_file:
                for attr in ['U', 'V', 'W', 'R', 'E', 't']:
                    data = getattr(meanFlow_to_save, attr)
                    if isinstance(data, np.ndarray):
                        data = np.squeeze(data).T
                    if attr != 't':
                        hdf5_file.create_dataset(attr, data=data, compression="gzip", compression_opts=9) #compression opts goes from 1 to 9 (1 is the less compressed)
                    else:
                        hdf5_file.create_dataset(attr, data=data)
    else:
        print(f'Resuming from file number {p.time.nStep}')

    # Write in log2.txt
    logFilePath = os.path.join(bin_path, 'log2.txt')
    with open(logFilePath, 'a') as logFile2:
        logFile2.write(f'DNS started at {datetime.now().strftime("%d-%b-%Y %H:%M:%S")}\n')
        logFile2.write(f'Parameters file: {caseFile}\n')
        logFile2.write(f'Starting flow file: flow_{p.time.nStep}.npy\n\n')

        if os.path.exists(os.path.join(bin_path, 'parameters.py')):
            parametersDiffStatus, parametersDiff = subprocess.getstatusoutput(f'diff {bin_path}/parameters.m {caseFile}.m')
            if parametersDiffStatus:
                logFile2.write(f'Parameters file was changed:\n{parametersDiff}\n')


    if extraParameters is not None:
        np.save(os.path.join(bin_path, 'extraParameters.npy'), extraParameters)

    # Compile Fortran code if needed
    if compileCode:
        print(f"Compiling code - {bin_path}")
        compileFortran(bin_path, decompDir=decompDir, optimizeCode=optimizeCode, debugger=debugger, profiler=profiler, displayCompiling=displayCompiling)  # Assumindo que compileFortran está definido em outro lugar

    # Plot the Domain if needed
    if plotDNSDomain:
        plotDomain() 
        import matplotlib.pyplot as plt
        plt.draw()

    # Call Fortran code
    if runSimulation and not debugger and not profiler:
        print('Starting code')
        start_time = datetime.now() 
        print(f'Command: cd {bin_path} && mpirun --allow-run-as-root -np {p_row*p_col} main {caseName}')
        subprocess.call(f'cd {bin_path} && mpirun --allow-run-as-root -np {p_row*p_col} main {caseName}', shell=True)
        print(f'Simulation completed in {(datetime.now() - start_time).seconds:.2f} seconds')
    elif runSimulation and debugger:
        print('Starting code with debugger')
        subprocess.call(f'cd {bin_path} && mpirun -n {p_row*p_col} xterm -sl 1000000 -fg white -bg black -hold -e gdb -ex run --args ./main {caseName}', shell=True)
    elif runSimulation and profiler:
        print('Starting code with profiler')
        os.environ['GMON_OUT_PREFIX'] = 'gmon.out'
        start_time = datetime.now() 
        subprocess.call(f'cd {bin_path} && mpirun -np {p_row*p_col} main {caseName}', shell=True)
        print(f'Simulation with profiler completed in {(datetime.now() - start_time).seconds:.2f} seconds')
        subprocess.call(f'cd {bin_path} && gprof -l main gmon.out > profile.txt', shell=True)
        shutil.move(os.path.join(bin_path, 'profile.txt'), '.')

    
    with open(logFilePath, 'a') as logFile2:
        logFile2.write(f'DNS finished at {datetime.now().strftime("%d-%b-%Y %H:%M:%S")}\n\n')

    flowHandles = []
    if 'flowHandles' in locals():
        allCaseFiles = os.listdir(case_path)
        for fileName in allCaseFiles:
            if len(fileName) == 19 and 'flow_' in fileName and fileName.endswith('.mat'):
                flowHandles.append(np.load(os.path.join(case_path, fileName), allow_pickle=True))

    info = {}
    if 'info' in locals():
        for varName in dir():
            info[varName] = eval(varName)

    

def unpackStruct(structure):
    varList = []

    def unpack(structure, parent_key=""):
        for varName, value in structure.__dict__.items():
            full_key = f"{parent_key}.{varName}" if parent_key else varName
            if isinstance(value, (structure, dict)):
                unpack(value, full_key)
            else:
                varList.append(full_key)

    unpack(structure)
    return varList

def compileFortran(bin_path, decompDir='/usr/local/2decomp_fft', optimizeCode=False, debugger=False, profiler=False, displayCompiling=True):
    """
    Compile Fortran files according to input options.
    
    Args:
        bin_path (str):
        decompDir (str):
        optimizeCode (bool):
        debugger (bool):
        profiler (bool):
        displayCompiling (bool):
    """

    makefile_extra_path = os.path.join(bin_path, 'makefile_extra')
    with open(makefile_extra_path, 'w') as out_file:
        out_file.write(f'DECOMPDIR = {decompDir}\n')

        if optimizeCode and not debugger:
            out_file.write('ARGS += -O5 -fcheck=all -fno-finite-math-only -march=native\n')
        
        if debugger:
            out_file.write('ARGS += -O0 -g -fbounds-check\n')
        elif profiler:
            out_file.write('ARGS += -g -pg\n')

    supress_output = '' if displayCompiling else ' >/dev/null'

    # Remove binary files to force the code to recompile
    main_binary_path = os.path.join(bin_path, 'main')
    if os.path.exists(main_binary_path):
        os.remove(main_binary_path)

    # Execute make
    command = f'cd {bin_path} && make --makefile=../../../source/Fortran/makefile {supress_output}'
    print(f'Command: {command}')
    status = os.system(command)

    # Verify the compilation status
    if status != 0:
        raise RuntimeError('Fortran compiling has failed')
    
def plotDomain(boundary, mesh):
    """
    Plots the domain before the execution.
    
    Args:
        boundary (BoundaryCondition):
        mesh (Mesh):
    """
    colors = plt.cm.get_cmap('tab10', 6).colors  
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    
    walls = [
        boundary.wall.front,
        boundary.wall.back,
        boundary.wall.up,
        boundary.wall.down,
        boundary.wall.right,
        boundary.wall.left
    ]
    
    for i, wallToPlot in enumerate(walls):
        if wallToPlot is not None and len(wallToPlot) > 0:
            wallToPlot = np.array(wallToPlot)
            if i in [0, 1]:  # front, back
                X = mesh.X[wallToPlot[:, [0, 0, 0, 0]]]
                Y = mesh.Y[wallToPlot[:, [2, 3, 3, 2]]]
                Z = mesh.Z[wallToPlot[:, [4, 4, 5, 5]]]
            elif i in [2, 3]:  # up, down
                X = mesh.X[wallToPlot[:, [0, 1, 1, 0]]]
                Y = mesh.Y[wallToPlot[:, [2, 2, 2, 2]]]
                Z = mesh.Z[wallToPlot[:, [4, 4, 5, 5]]]
            elif i in [4, 5]:  # right, left
                X = mesh.X[wallToPlot[:, [0, 1, 1, 0]]]
                Y = mesh.Y[wallToPlot[:, [2, 2, 3, 3]]]
                Z = mesh.Z[wallToPlot[:, [4, 4, 4, 4]]]
            
            for j in range(X.shape[0]):
                verts = [list(zip(X[j, :], Z[j, :], Y[j, :]))]
                poly = Poly3DCollection(verts, color=colors[i], edgecolor='k')
                ax.add_collection3d(poly)
    
    corners = np.array(boundary.corners.limits)
    cornerDir = np.array(boundary.corners.dir)
    
    for i in range(corners.shape[0]):
        if np.sum(np.abs(cornerDir[i, :])) == 2:
            ax.plot3D(mesh.X[corners[i, [0, 1]]], 
                      mesh.Z[corners[i, [4, 5]]], 
                      mesh.Y[corners[i, [2, 3]]], 
                      'r', linewidth=2)
        else:
            ax.plot3D([mesh.X[corners[i, 0]]], 
                      [mesh.Z[corners[i, 4]]], 
                      [mesh.Y[corners[i, 2]]], 
                      'ro')

        X_mean = np.mean(mesh.X[corners[i, [0, 1]]])
        Y_mean = np.mean(mesh.Y[corners[i, [2, 3]]])
        Z_mean = np.mean(mesh.Z[corners[i, [4, 5]]])
        
        scale = 0.1
        ax.plot3D([X_mean, X_mean + cornerDir[i, 0] * scale],
                  [Z_mean, Z_mean + cornerDir[i, 2] * scale],
                  [Y_mean, Y_mean + cornerDir[i, 1] * scale],
                  'g', linewidth=2)
    
    ax.set_aspect('auto')
    ax.autoscale_view()
    ax.view_init(elev=30, azim=30)
    plt.show()
