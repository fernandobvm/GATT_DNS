% Script em MATLAB para testar a função calcCompressibleBL

% Parâmetros do fluxo
flowParameters.Re = 1e6;       % Número de Reynolds
flowParameters.Ma = 0.3;       % Número de Mach
flowParameters.Pr = 0.71;      % Número de Prandtl
flowParameters.T0 = 300;       % Temperatura [K]
flowParameters.gamma = 1.4;    % Gamma

adiabWall = true; % Parede adiabática

% Malha
mesh.X = linspace(0, 10, 100); % Distribuição em X
mesh.Y = linspace(-5, 5, 100); % Distribuição em Y
mesh.Z = linspace(0, 10, 50);  % Distribuição em Z

% Chamar a função para calcular a camada limite compressível
flow = calcCompressibleBL(flowParameters, adiabWall, mesh);

% Exibir resultados para comparação
disp('Resultados MATLAB:')
disp(flow.U) % Velocidade U
disp(flow.R) % Densidade R
