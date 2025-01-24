import numpy as np
import os
import h5py

def func_loadFlow(case_path, ax_lim, var_list, steps=None, d_step=None, n_step=None):
    print(f" Loading flow...\n   case path: {case_path}")

    if steps is None:
        steps = func_findSteps(case_path, d_step, n_step)  # Function assumed to be implemented externally

    if len(steps) > 1:
        print(f"   qSave: {steps[0]}:{steps[1] - steps[0]}:{steps[-1]} ({len(steps)} steps)")
    else:
        print(f"   qSave: {steps[0]}")

    # Load mesh data
    mesh = np.load(os.path.join(case_path, 'mesh.npz'))
    x = mesh['X']
    y = mesh['Y']

    ax_lim = np.where(np.isinf(ax_lim), 1e6 * np.sign(ax_lim), ax_lim)
    ix = np.where((x >= ax_lim[0]) & (x <= ax_lim[1]))[0]
    iy = np.where((y >= ax_lim[2]) & (y <= ax_lim[3]))[0]

    x = x[ix]
    y = y[iy]
    print(f"   x: [{x[0]:.2f}, {x[-1]:.2f}]")
    print(f"   y: [{y[0]:.2f}, {y[-1]:.2f}]")

    n_steps = len(steps)
    t = np.zeros(n_steps)
    var_out = {var: np.zeros((len(x), len(y), n_steps)) for var in var_list}

    for i_step, step in enumerate(steps):
        with h5py.File(os.path.join(case_path, f'flow_{step:010d}.h5'), 'r') as flow_data:
            for var in var_list:
                data = flow_data[var][...]
                var_out[var][:, :, i_step] = data[np.ix_(ix, iy)]

    return x, y, t, var_out

def func_findSteps(path_case, d_step=None, n_step=None):
    if d_step is None:
        d_step = 1
    if n_step is None:
        n_step = 1

    all_files = os.listdir(path_case)
    all_steps = []
    for file_name in all_files:
        if file_name.startswith('flow_'):
            step = int(file_name[5:-3])  # Extract numeric part between 'flow_' and '.h5'
            all_steps.append(step)

    all_steps = sorted(all_steps)

    if n_step > 1:
        i_steps = list(range(len(all_steps) - 1, -1, -d_step))[:n_step]
        i_steps = sorted(i_steps)
    else:
        i_steps = [len(all_steps) - 1]

    steps = [all_steps[i] for i in i_steps]

    return steps
