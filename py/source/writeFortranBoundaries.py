class BoundaryInfoWriter:
    def __init__(self, case_name, bi):
        self.case_name = case_name
        self.bi = bi
        self.out_file = None

    def write_boundary_info(self):
        n_procs = len(self.bi)
        self.out_file = open(f"{self.case_name}/bin/boundaryInfo.F90", 'w')
        self.out_file.write("    select case (nrank)\n")

        vars = ['U', 'V', 'W', 'E', 'P']

        for i in range(n_procs):
            self.out_file.write(f"        case ({i})\n")
            self.write_dirichlet_conditions(i, vars)

        self.out_file.write("    end select\n")
        self.out_file.close()

    def write_dirichlet_conditions(self, i, vars):
        for j, var in enumerate(vars):
            if j == 0:
                n, ind, val = self.bi[i].nUd, self.bi[i].iUd, self.bi[i].vUd
            elif j == 1:
                n, ind, val = self.bi[i].nVd, self.bi[i].iVd, self.bi[i].vVd
            elif j == 2:
                n, ind, val = self.bi[i].nWd, self.bi[i].iWd, self.bi[i].vWd
            elif j == 3:
                n, ind, val = self.bi[i].nEd, self.bi[i].iEd, self.bi[i].vEd
            elif j == 4:
                n, ind, val = self.bi[i].nPd, self.bi[i].iPd, self.bi[i].vPd

            # Aqui a lógica para escrever `n`, `ind`, `val` no arquivo segue
            self.out_file.write(f"        ! {var} Dirichlet conditions\n")
            # Escreve as condições de contorno, supondo que `n`, `ind`, `val` sejam iteráveis
            self.out_file.write(f"        n_{var.lower()} = {n}\n")
            self.out_file.write(f"        indices_{var.lower()} = {ind}\n")
            self.out_file.write(f"        values_{var.lower()} = {val}\n")

# Exemplo de uso
bi = [  # Definir os dados da classe bi como apropriado
    # Objeto com atributos nUd, iUd, vUd, nVd, iVd, etc.
]

writer = BoundaryInfoWriter("case_name", bi)
writer.write_boundary_info()
