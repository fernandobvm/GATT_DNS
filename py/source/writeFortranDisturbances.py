import os

class DisturbanceWriter:
    def __init__(self, case_name, bi, tridimensional):
        self.case_name = case_name
        self.bi = bi
        self.tridimensional = tridimensional
        self.disturb_types = []

    def write_files(self):
        n_procs = len(self.bi)

        with open(f"{self.case_name}/bin/runDisturbances.F90", 'w') as out_file_disturb:
            if not self.tridimensional:
                with open(f"{self.case_name}/bin/runForcings2D.F90", 'w') as out_file_forcing:
                    os.system(f"touch {self.case_name}/bin/runForcings3D.F90")
            else:
                with open(f"{self.case_name}/bin/runForcings3D.F90", 'w') as out_file_forcing:
                    os.system(f"touch {self.case_name}/bin/runForcings2D.F90")
            
            out_file_disturb.write("    select case (nrank)\n")
            out_file_forcing.write("    select case (nrank)\n")

            for i in range(n_procs):
                out_file_disturb.write(f"        case ({i})\n")
                out_file_forcing.write(f"        case ({i})\n")
                self.write_disturbances(i, out_file_disturb, out_file_forcing)

            out_file_forcing.write("    end select\n")
            out_file_disturb.write("    end select\n")

        self.disturb_types = list(set(self.disturb_types))  # Remove duplicates
        self.write_disturbances_module()

    def write_disturbances(self, i, out_file_disturb, out_file_forcing):
        for j, di in enumerate(self.bi[i].disturb):
            if di.forcing:
                out_file = out_file_forcing
            else:
                out_file = out_file_disturb
            
            self.disturb_types.append(di.type)

            nx = di.ind[1] - di.ind[0] + 1
            ny = di.ind[3] - di.ind[2] + 1
            nz = di.ind[5] - di.ind[4] + 1

            out_file.write(f"            call {di.type}({nx},{ny},{nz},(/")

            # Writing X coordinates
            for k in range(nx - 1):
                out_file.write(f"{di.X[k]:.20f}d0,")
            out_file.write(f"{di.X[-1]:.20f}d0/),(/")

            # Writing Y coordinates
            for k in range(ny - 1):
                out_file.write(f"{di.Y[k]:.20f}d0,")
            out_file.write(f"{di.Y[-1]:.20f}d0/),(/")

            # Writing Z coordinates
            for k in range(nz - 1):
                out_file.write(f"{di.Z[k]:.20f}d0,")
            if di.forcing:
                out_file.write(f"{di.Z[-1]:.20f}d0/)")
            else:
                out_file.write(f"{di.Z[-1]:.20f}d0/),t")

            # Writing variable names and indices
            for var in di.var:
                out_file.write(f",{var}({di.ind[0]}:{di.ind[1]},{di.ind[2]}:{di.ind[3]},{di.ind[4]}:{di.ind[5]})")
            
            if di.forcing:
                for var in di.var:
                    out_file.write(f",d{var}({di.ind[0]}:{di.ind[1]},{di.ind[2]}:{di.ind[3]},{di.ind[4]}:{di.ind[5]})")
            
            # Writing additional parameters
            for par in di.par:
                if isinstance(par, (int, float)):  # Single number
                    out_file.write(f",{par:.20f}d0")
                elif isinstance(par, (list, tuple)):  # Vector
                    out_file.write(",(/")
                    for p in par[:-1]:
                        out_file.write(f"{p:.20f}d0,")
                    out_file.write(f"{par[-1]:.20f}d0/)")
                else:  # String
                    out_file.write(f",'{par}'")

            out_file.write(")\n")

    def write_disturbances_module(self):
        with open(f"{self.case_name}/bin/disturbances.F90", 'w') as out_file:
            out_file.write("    module disturbances\n\n    contains\n\n")
            
            for disturb_type in self.disturb_types:
                source_file_path = f"source/disturbances/{disturb_type}.F90" if disturb_type != 'holdInlet' else f"source/Fortran/{disturb_type}.F90"
                with open(source_file_path, 'r') as source_file:
                    for line in source_file:
                        out_file.write(f"{line}\n")
                out_file.write("\n")

            out_file.write("\n    end module\n")

# Exemplo de uso:
# Assumindo que `bi` seja uma lista de objetos que tenham as propriedades `disturb`, `forcing`, `type`, `ind`, `X`, `Y`, `Z`, `var`, `par`.
bi = [
    # Objetos de exemplo com os atributos apropriados
]

# Inicializando e executando a escrita dos arquivos
writer = DisturbanceWriter("case_name", bi, tridimensional=True)
writer.write_files()
