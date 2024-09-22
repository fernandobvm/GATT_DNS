import os
import subprocess

class FortranCompiler:
    def __init__(self, case_name, decomp_dir, optimize_code, debugger, profiler, display_compiling, matlab_dir=None):
        self.case_name = case_name
        self.decomp_dir = decomp_dir
        self.optimize_code = optimize_code
        self.debugger = debugger
        self.profiler = profiler
        self.display_compiling = display_compiling
        self.matlab_dir = matlab_dir if matlab_dir else os.getenv('MATLABROOT')  # Use environment variable if not provided

    def create_makefile(self):
        # Create extra makefile with the directories that are specific to this run
        makefile_path = os.path.join(self.case_name, 'bin', 'makefile_extra')
        with open(makefile_path, 'w') as out_file:
            out_file.write(f'MATROOT = {self.matlab_dir}\n')
            out_file.write(f'DECOMPDIR = {self.decomp_dir}\n')

            if self.optimize_code and not self.debugger:  # Optimization options
                out_file.write('ARGS += -O5 -fcheck=all -fno-finite-math-only -march=native\n')

            if self.debugger:  # Debugging options
                out_file.write('ARGS += -O0 -g -fbounds-check\n')
            elif self.profiler:  # Profiling options
                out_file.write('ARGS += -g -pg\n')

    def compile(self):
        # Run make
        self.create_makefile()

        suppress_output = '' if self.display_compiling else ' > /dev/null'  # Suppress the compiler output

        main_binary_path = os.path.join(self.case_name, 'bin', 'main')
        if os.path.isfile(main_binary_path):  # Remove the main binary to force recompiling
            os.remove(main_binary_path)

        # Call make
        try:
            subprocess.run(['make', '--makefile=../../source/Fortran/makefile'], cwd=os.path.join(self.case_name, 'bin'), check=True)
        except subprocess.CalledProcessError:
            raise RuntimeError('Fortran compiling has failed')
