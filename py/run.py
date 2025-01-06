import parameters
from runDNS import *
from source.MatricesFiles import *
from source.BoundaryFiles import *
from source.preprocessing import *

#[case_name, flow_params, domain, flow_type, mesh, time, num_methods, p_row, p_col] = parameters()

caseFile = 'parameters'

[flowHandles, info] = runDNS()