import os
import re
import numpy as np

class PreviousRunChecker:
    def __init__(self, case_name):
        self.case_name = case_name

    def check_previous_run(self):
        # This function checks for previous run files
        # case_name is the folder to be checked
        # nStep is the last time step found
        # nx, ny, and nz are the size of the mesh in the saved file
        # If no files are found, empty arrays are returned

        all_files = os.listdir(self.case_name)  # List all files

        case_files = []  # Flow file will be placed here

        for name in all_files:
            if len(name) == 19 and re.search(r'flow_\d*\.npy', name):  # Check the file name
                case_files.append(name)

        if not case_files:
            return None, None, None, None

        n_steps = [int(re.findall(r'\d+', name)[0]) for name in case_files]
        n_step = max(n_steps)

        # If nx, ny, nz are requested
        file_path = os.path.join(self.case_name, f'flow_{n_step:010d}.npy')
        data = np.load(file_path)  # Load the .npy file
        nx, ny, nz = data.shape

        return n_step, nx, ny, nz
