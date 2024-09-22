import numpy as np

class Matrices:
    def __init__(self, x, y, z=None, neumann_coeffs=None, neumann2_coeffs=None):
        self.x = x
        self.y = y
        self.z = z
        self.neumann_coeffs = neumann_coeffs or []
        self.neumann2_coeffs = neumann2_coeffs or []

class Mesh:
    def __init__(self, tracked_points=None, X=None, Y=None, Z=None, x=None, y=None, z=None):
        self.tracked_points = tracked_points
        self.X = X
        self.Y = Y
        self.Z = Z
        self.x = x
        self.y = y
        self.z = z

def write_matrix(out_file, name, var):
    out_file.write(f"    {name} = reshape((/")
    for i, value in enumerate(var.flatten()[:-1]):
        out_file.write(f"{value:.20f}d0,")
    out_file.write(f"{var.flatten()[-1]:.20f}d0")
    out_file.write(f"/),shape({name}))\n")

def write_fortran_matrices(case_name, matrices, num_methods, mesh):
    with open(f"{case_name}/bin/matrices.F90", "w") as out_file:
        I, J = matrices.y.types.shape[0], matrices.x.types.shape[0]
        K = matrices.x.types.shape[1]
        n_procs = len(matrices.x.blocks)

        # Write blocks
        out_file.write("    select case(nrank)\n")
        for i in range(n_procs):
            out_file.write(f"        case({i})\n")
            
            # For X
            blocks = matrices.x.blocks[i]
            n_blocks = blocks.shape[0]
            out_file.write(f"            nDerivBlocksX = {n_blocks}\n")
            out_file.write(f"            allocate(derivBlocksX({n_blocks},5))\n")
            out_file.write("            derivBlocksX = reshape((/")
            for n in range(n_blocks*5-1):
                out_file.write(f"{blocks.flatten()[n]},")
            out_file.write(f"{blocks.flatten()[-1]}")
            out_file.write("/),shape(derivBlocksX))\n\n")

            # For Y
            blocks = matrices.y.blocks[i]
            n_blocks = blocks.shape[0]
            out_file.write(f"            nDerivBlocksY = {n_blocks}\n")
            out_file.write(f"            allocate(derivBlocksY({n_blocks},5))\n")
            out_file.write("            derivBlocksY = reshape((/")
            for n in range(n_blocks*5-1):
                out_file.write(f"{blocks.flatten()[n]},")
            out_file.write(f"{blocks.flatten()[-1]}")
            out_file.write("/),shape(derivBlocksY))\n\n")

            # For Z (if applicable)
            if K > 1:
                blocks = matrices.z.blocks[i]
                n_blocks = blocks.shape[0]
                out_file.write(f"            nDerivBlocksZ = {n_blocks}\n")
                out_file.write(f"            allocate(derivBlocksZ({n_blocks},5))\n")
                out_file.write("            derivBlocksZ = reshape((/")
                for n in range(n_blocks*5-1):
                    out_file.write(f"{blocks.flatten()[n]},")
                out_file.write(f"{blocks.flatten()[-1]}")
                out_file.write("/),shape(derivBlocksZ))\n\n")

        out_file.write("    end select\n\n")

        # Filter info
        out_file.write(f"    filterX = {num_methods.filter_directions[0]}\n")
        out_file.write(f"    filterY = {num_methods.filter_directions[1]}\n")
        if K > 1:
            out_file.write(f"    filterZ = {num_methods.filter_directions[2]}\n")

        # X derivatives
        out_file.write(f"    derivnRHSx = {matrices.x.n_rhs}\n")
        out_file.write(f"    filternRHSx = {matrices.x.n_rhsf}\n")
        out_file.write(f"    allocate(derivsAX({I-1},{matrices.x.n_types}))\n")
        out_file.write(f"    allocate(derivsBX({I-matrices.x.periodic},{matrices.x.n_types}))\n")
        write_matrix(out_file, 'derivsAX', matrices.x.A)
        write_matrix(out_file, 'derivsBX', matrices.x.B)
        out_file.write(f"    periodicX = {matrices.x.periodic}\n")

        # Repeat similar sections for Y, Z, and filters...
