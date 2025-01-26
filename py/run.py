import os
import sys
import numpy as np
from source.library import *
from Simulation import runDNS

if len(sys.argv) < 2:
    print("Por favor, forneça o nome do arquivo de parâmetros (sem extensão).")
    sys.exit(1)

with open(f'{sys.argv[1]}.py', "r") as f:
    python_code = f.read()

exec(python_code)

runDNS(case_name, flow_params, domain, flow_type, mesh, time, num_methods, p_row, p_col, caseFile, logAll)