import os
import re
import shutil
import numpy as np
import h5py

class MeanFlowCalculator:
    def __init__(self, case_path, nq_save):
        self.case_path = case_path
        self.nq_save = nq_save
    
    def calc_meanflow(self):
        all_files = os.listdir(self.case_path)
        case_files = [f for f in all_files if re.fullmatch(r'flow_\d{1,}.h5', f)]
        
        if not case_files:
            print('!!! WARNING: FLOW FILES NOT FOUND\n!!! CALC. NOT DONE\n')
            return
        
        n_steps_full = sorted(int(re.search(r'\d+', f).group()) for f in case_files)
        if n_steps_full[0] == 0:
            n_steps_full = n_steps_full[1:]
        
        if len(n_steps_full) < self.nq_save:
            print(f'!!! WARNING: NqSave > NSteps ({self.nq_save} > {len(n_steps_full)})\n!!! CALC. NOT DONE\n')
            return
        
        q_save_t = n_steps_full[-self.nq_save:]
        d_q_save = np.unique(np.diff(q_save_t))
        
        if len(d_q_save) != 1:
            if len(d_q_save) == 0:
                print('Warning: qSave is empty...')
                return
            else:
                print('Warning: qSave is not constant...')
                return
        
        print(f'[MEAN FLOW CALCULATION]\n  CASE PATH : {self.case_path}\n  FLOW FILES: {q_save_t[0]}:{d_q_save[0]}:{q_save_t[-1]}\n  NUMB FILES: {len(q_save_t)}\n')
        self._calc_meanflow(q_save_t)
        print('[FINISHED]\n')
    
    def _calc_meanflow(self, tin):
        tout = tin[-1]
        pasta_in = self.case_path
        
        old_flow_path = os.path.join(pasta_in, f'old_flow_{tout:010d}.h5')
        new_flow_path = os.path.join(pasta_in, f'flow_{tout:010d}.h5')
        
        if os.path.exists(new_flow_path):
            shutil.copy(new_flow_path, old_flow_path)
        
        u, v, w, r, e = None, None, None, None, None
        
        for t in tin:
            tstr = os.path.join(pasta_in, f'flow_{t:010d}.h5')
            with h5py.File(tstr, 'r') as f:
                if u is None:
                    u, v, w, r, e = f['U'][:], f['V'][:], f['W'][:], f['R'][:], f['E'][:]
                else:
                    u += f['U'][:]
                    v += f['V'][:]
                    w += f['W'][:]
                    r += f['R'][:]
                    e += f['E'][:]
        
        u /= len(tin)
        v /= len(tin)
        w /= len(tin)
        r /= len(tin)
        e /= len(tin)
        
        with h5py.File(new_flow_path, 'w') as f:
            f.create_dataset('U', data=u)
            f.create_dataset('V', data=v)
            f.create_dataset('W', data=w)
            f.create_dataset('R', data=r)
            f.create_dataset('E', data=e)
        
        meanflow_path = os.path.join(pasta_in, 'meanflowSFD.h5')
        old_meanflow_path = os.path.join(pasta_in, f'old_meanflowSFD_{tout:010d}.h5')
        
        if os.path.exists(meanflow_path):
            shutil.move(meanflow_path, old_meanflow_path)

# Uso
case_path = './baseflow_gap_Re950_M01_h10_LR18_case2'
nq_save = 122
calculator = MeanFlowCalculator(case_path, nq_save)
calculator.calc_meanflow()
