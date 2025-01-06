import os
import shutil
import subprocess
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from datetime import datetime
from source.library import *
from parameters import *
from source.MatricesFiles import *
from source.BoundaryFiles import *
from source.preprocessing import *

def runDNS(extraParameters = None, caseFile = 'parameters'):

    # Define se a simulação será realmente compilada e executada ou apenas o pré-processamento será feito
    runSimulation = True
    compileCode = True
    plotDNSDomain = False

    # Parâmetros de compilação
    forceRecompileAll = False
    displayCompiling = False
    optimizeCode = True
    debugger = False
    profiler = False

    # Configuração das pastas de bibliotecas
    matlabDir = ''  # Deixe vazio para diretório automático
    decompDir = '/usr/local/2decomp_fft'

    # Registro de dados
    logAll = False  # Salvar todas as iterações no log ou apenas quando um fluxo é salvo

    # Executa o arquivo de parâmetros
    #eval(caseFile)
    if caseFile == 'parameters':
        [caseName, flowParameters, domain, flowType, mesh, time, numMethods, p_row, p_col, logAll] = parameters()

    if extraParameters is not None:
        # Descompacta extraParameters e define variáveis adicionais
        extraVars = unpackStruct(extraParameters)
        for var in extraVars:
            exec(f"{var} = extraParameters['{var}']")

    # Verifica se é 2D ou 3D
    tridimensional = mesh.z.n + mesh.z.buffer_i.n + mesh.z.buffer_f.n > 1
    if not tridimensional:
        p_col = 1

    # Adiciona caminho para o código fonte
    source_path = 'source'
    boundaries_path = 'source/boundaries'
    if source_path not in os.sys.path:
        os.sys.path.append(source_path)
    if boundaries_path not in os.sys.path:
        os.sys.path.append(boundaries_path)

    # Inicializa o arquivo de log
    if not os.path.exists(caseName):
        os.mkdir(caseName)

    print(f'Parameters file: {caseFile}\nCase name: {caseName}')

    log_file_path = os.path.join(caseName, 'log.txt')
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as logFile:
            logFile.write('Save number\tIteration\tSimulation time\tdt       \tCFL      \tU change\tV change\tW change\tR change\tE change')
            if hasattr(mesh, 'trackedPoints'):
                for i in range(mesh.trackedPoints.shape[0]):
                    logFile.write(f'\tU{i+1}\tV{i+1}\tW{i+1}\tR{i+1}\tE{i+1}')
            logFile.write('\n')

    # Executa rotina de pré-processamento ou recarrega anterior
    bin_path = os.path.join(caseName, 'bin')
    if not os.path.exists(bin_path):
        os.mkdir(bin_path)

    if forceRecompileAll:
        compileCode = True
        for ext in ['.mod', '.o']:
            for file in os.listdir(bin_path):
                if file.endswith(ext):
                    os.remove(os.path.join(bin_path, file))

    print('Running preprocessor')
    p = Preprocessing(domain, mesh, flowType, flowParameters, numMethods, caseName, p_row, p_col, logAll, tridimensional, time)
    print(p.time.nStep)
    print(f'Mesh size: {mesh.nx} x {mesh.ny} x {mesh.nz}')

    # Gerar fluxo inicial, se necessário
    if p.genInitialFlow:  # Se não houver execução anterior, gerar novo fluxo inicial
        print('Generating new initial flow')

        # Computar fluxo inicial
        flow = generateInitialFlow(mesh, flowParameters, flowType, p.boundary.inside_wall, flowType.name)

        # Salvar fluxo inicial em arquivo
        flow_to_save = flow
        flow_to_save.t = 0
        for var in 'UVWRE':
            setattr(flow_to_save, var, np.where(p.boundary.inside_wall, np.nan, getattr(flow_to_save, var)))

        # Salvar fluxo inicial em formato .npy
        np.save(f'{caseName}/flow_0000000000.npy', flow_to_save)

        # Verificar se há arquivo de fluxo médio
        if flowType.initial_meanFile != None:
            flowTypeTemp = FlowType()  # Criar instância temporária
            flowTypeTemp.initial_type = 'file'
            flowTypeTemp.initial_flowFile = flowType.initial_meanFile
            if flowType.initial_meshFile != None:
                flowTypeTemp.initial_meshFile = flowType.initial_meshFile

            meanFlow = generateInitialFlow(mesh, flowParameters, flowTypeTemp, p.boundary.inside_wall, flowType.name)

            # Salvar fluxo médio em arquivo
            meanFlow_to_save = meanFlow
            meanFlow_to_save.t = 0
            for var in 'UVWRE':
                setattr(meanFlow_to_save, var, np.where(p.boundary.inside_wall, np.nan, getattr(meanFlow_to_save, var)))

            np.save(f'{caseName}/meanflowSFD.npy', meanFlow_to_save)

    else:
        print(f'Resuming from file number {p.time.nStep}')

    # Escrever no log2.txt
    logFilePath = os.path.join(caseName, 'bin', 'log2.txt')
    with open(logFilePath, 'a') as logFile2:
        logFile2.write(f'DNS started at {datetime.now().strftime("%d-%b-%Y %H:%M:%S")}\n')
        logFile2.write(f'Parameters file: {caseFile}\n')
        logFile2.write(f'Starting flow file: flow_{p.time.nStep}.npy\n\n')

        if os.path.exists(os.path.join(caseName, 'bin', 'parameters.py')):
            parametersDiffStatus, parametersDiff = subprocess.getstatusoutput(f'diff {caseName}/bin/parameters.m {caseFile}.m')
            if parametersDiffStatus:
                logFile2.write(f'Parameters file was changed:\n{parametersDiff}\n')

    # Copiar o arquivo de parâmetros para a pasta Fortran
    shutil.copyfile(f'{caseFile}.py', os.path.join(caseName, 'bin', 'parameters.py'))

    if extraParameters is not None:
        # Salvar extraParameters no formato .npy
        np.save(os.path.join(caseName, 'bin', 'extraParameters.npy'), extraParameters)

    # Compilar o código Fortran se necessário
    if compileCode:
        print('Compiling code')
        compileFortran(caseName)  # Assumindo que compileFortran está definido em outro lugar

    # Plotar o domínio se necessário
    if plotDNSDomain:
        plotDomain()  # Assumindo que plotDomain está definido em outro lugar
        import matplotlib.pyplot as plt
        plt.draw()

    # Chamar o código Fortran
    if runSimulation and not debugger and not profiler:
        print('Starting code')
        start_time = time.time()
        subprocess.call(f'cd {caseName}/bin && mpirun -np {p_row*p_col} main {caseName}', shell=True)
        print(f'Simulation completed in {time.time() - start_time:.2f} seconds')
    elif runSimulation and debugger:
        print('Starting code with debugger')
        subprocess.call(f'cd {caseName}/bin && mpirun -n {p_row*p_col} xterm -sl 1000000 -fg white -bg black -hold -e gdb -ex run --args ./main {caseName}', shell=True)
    elif runSimulation and profiler:
        print('Starting code with profiler')
        os.environ['GMON_OUT_PREFIX'] = 'gmon.out'
        start_time = time.time()
        subprocess.call(f'cd {caseName}/bin && mpirun -np {p_row*p_col} main {caseName}', shell=True)
        print(f'Simulation with profiler completed in {time.time() - start_time:.2f} seconds')
        subprocess.call(f'cd {caseName}/bin && gprof -l main gmon.out > profile.txt', shell=True)
        shutil.move(os.path.join(caseName, 'bin', 'profile.txt'), '.')

    # Escrever no log2.txt ao final
    with open(logFilePath, 'a') as logFile2:
        logFile2.write(f'DNS finished at {datetime.now().strftime("%d-%b-%Y %H:%M:%S")}\n\n')

    # Obter resultados, se necessário
    flowHandles = []
    if 'flowHandles' in locals():
        allCaseFiles = os.listdir(caseName)
        for fileName in allCaseFiles:
            if len(fileName) == 19 and 'flow_' in fileName and fileName.endswith('.mat'):
                flowHandles.append(np.load(os.path.join(caseName, fileName), allow_pickle=True))

    # Gerar estrutura de informação se necessário
    info = {}
    if 'info' in locals():
        for varName in dir():
            info[varName] = eval(varName)

    return flowHandles, info

def unpackStruct(structure):
    varList = []

    def unpack(structure, parent_key=""):
        # Obtém as variáveis dentro da estrutura
        for varName, value in structure.__dict__.items():
            full_key = f"{parent_key}.{varName}" if parent_key else varName
            if isinstance(value, (structure, dict)):  # Assume-se que subestruturas podem ser dicts ou classes
                unpack(value, full_key)
            else:
                varList.append(full_key)

    unpack(structure)
    return varList

def compileFortran(case_name, dir=None, decomp_dir=None, optimize_code=False, debugger=False, profiler=False, display_compiling=True):
    """
    Compila arquivos Fortran de acordo com as opções fornecidas.
    
    Args:
        case_name (str): Nome do caso/pasta onde o makefile será gerado.
        dir (str): Diretório do Matlab. Se None, usa o valor padrão.
        decomp_dir (str): Diretório de decomposição.
        optimize_code (bool): Se True, ativa otimizações de código.
        debugger (bool): Se True, ativa opções de depuração.
        profiler (bool): Se True, ativa opções de profiling.
        display_compiling (bool): Se False, suprime a saída de compilação.
    """
    # Se o diretório do Matlab não for fornecido, use o valor padrão
    if dir is None:
        dir = os.getcwd()  # Use o valor do sistema ou um default

    # Cria o arquivo makefile_extra
    makefile_extra_path = os.path.join(case_name, 'bin', 'makefile_extra')
    with open(makefile_extra_path, 'w') as out_file:
        out_file.write(f'MATROOT = {dir}\n')
        out_file.write(f'DECOMPDIR = {decomp_dir}\n')

        # Adiciona as opções de otimização ou depuração
        if optimize_code and not debugger:
            out_file.write('ARGS += -O5 -fcheck=all -fno-finite-math-only -march=native\n')
        
        if debugger:
            out_file.write('ARGS += -O0 -g -fbounds-check\n')
        elif profiler:
            out_file.write('ARGS += -g -pg\n')

    # Verifica se deve suprimir a saída da compilação
    supress_output = '' if display_compiling else ' >/dev/null'

    # Remove o arquivo binário main se ele já existir, para forçar a recompilação
    main_binary_path = os.path.join(case_name, 'bin', 'main')
    if os.path.exists(main_binary_path):
        os.remove(main_binary_path)

    # Executa o comando make
    command = f'cd {os.path.join(case_name, "bin")} && make --makefile=../../source/Fortran/makefile {supress_output}'
    status = os.system(command)

    # Verifica se a compilação foi bem-sucedida
    if status != 0:
        raise RuntimeError('Fortran compiling has failed')
    
def plotDomain(boundary, mesh):
    """
    Plota o domínio antes do tempo de execução.
    
    Args:
        boundary (BoundaryCondition): Instância da classe BoundaryCondition contendo as informações de limites.
        mesh (Mesh): Instância da classe Mesh contendo os dados do malha.
    """
    colors = plt.cm.get_cmap('tab10', 6).colors  # Usar esquema de cores similar ao 'lines' do Matlab
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    
    # Mapeamento das faces
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
            
            # Desenhar as paredes
            for j in range(X.shape[0]):
                verts = [list(zip(X[j, :], Z[j, :], Y[j, :]))]
                poly = Poly3DCollection(verts, color=colors[i], edgecolor='k')
                ax.add_collection3d(poly)
    
    # Desenho dos corners
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