import numpy as np
from scipy.integrate import trapezoid as trapz
from scipy.interpolate import interp1d
#TODO: Revisar a implementação e os testes gerados pelo gpt
class CompressibleBLFlow:
    def __init__(self, flow_params, adiabatic_wall, mesh):
        self.Re = flow_params.Re
        self.Minf = flow_params.Ma
        self.Pr = flow_params.Pr
        self.Tinf = flow_params.T0
        self.gamma = flow_params.gamma
        self.Twall = flow_params.Twall / self.Tinf if hasattr(flow_params, 'Twall') else 1

        self.adiabatic_wall = adiabatic_wall
        self.mesh = mesh
        self.xR = self.Re / 1.7208**2
        self.E0 = 1 / (self.gamma * (self.gamma - 1) * self.Minf**2)

        # Solve the boundary layer equations
        self.solution = self.solve_compressible_bl()

        # Interpolate the results to the mesh grid
        self.U, self.V, self.R, self.E = self.interpolate_to_mesh()

    def solve_compressible_bl(self):
        return SolveCompressibleBL(self.Minf, self.Pr, self.Tinf, self.Twall, self.gamma, self.adiabatic_wall)

    def interpolate_to_mesh(self):
        X = self.mesh.X
        Y = self.mesh.Y
        Z = self.mesh.Z

        nx, ny, nz = len(X), len(Y), len(Z)
        U = np.ones((nx, ny))
        V = np.zeros((nx, ny))
        R = np.ones((nx, ny))
        E = np.ones((nx, ny)) * self.E0

        eta_xR = self.solution.eta
        U_xR = self.solution.f_p
        R_xR = 1.0 / self.solution.rbar
        E_xR = self.solution.rbar * self.E0
        V_xR = -(self.solution.rbar) * (self.solution.f - eta_xR * self.solution.f_p) * (1.7208 / np.sqrt(2)) * (1 / self.Re)

        y_xR = eta_xR * np.sqrt(2) / 1.7208

        dS_xR = trapz(1 - U_xR * R_xR / (U_xR[-1] * R_xR[-1]), y_xR)

        print(f"Initial Flow: Compressible BL")
        if self.adiabatic_wall:
            print(f"   Adiabatic Wall, (dT/dy)_wall = 0, (Twall/TInf) = {E_xR[0] / self.E0:.3f}")
        else:
            print(f"   Isothermal Wall, T_wall = const, (Twall/TInf) = {E_xR[0] / self.E0:.3f}")
        print(f"   !!! BL thickness, (effective) / (incomp. BL, Blasius) = {dS_xR:.4f}")

        for ix in range(len(X)):
            if X[ix] > 0:
                y_xL = y_xR * np.sqrt(X[ix] / self.xR)

                U[ix, :] = interp1d(y_xL, U_xR, kind='makima', fill_value=U_xR[-1], bounds_error=False)(Y)
                V[ix, :] = interp1d(y_xL, V_xR, kind='makima', fill_value=V_xR[-1], bounds_error=False)(Y)
                R[ix, :] = interp1d(y_xL, R_xR, kind='makima', fill_value=R_xR[-1], bounds_error=False)(Y)
                E[ix, :] = interp1d(y_xL, E_xR, kind='makima', fill_value=E_xR[-1], bounds_error=False)(Y)

        # Set U, V, R, E at the origin and for negative Y values
        U[X == 0, Y == 0] = 0
        U[:, Y < 0] = np.tile(U[:, Y == 0], (1, len(Y[Y < 0])))
        V[:, Y < 0] = np.tile(V[:, Y == 0], (1, len(Y[Y < 0])))
        R[:, Y < 0] = np.tile(R[:, Y == 0], (1, len(Y[Y < 0])))
        E[:, Y < 0] = np.tile(E[:, Y == 0], (1, len(Y[Y < 0])))

        U = np.repeat(U[:, :, np.newaxis], nz, axis=2)
        V = np.repeat(V[:, :, np.newaxis], nz, axis=2)
        R = np.repeat(R[:, :, np.newaxis], nz, axis=2)
        E = np.repeat(E[:, :, np.newaxis], nz, axis=2)

        return U, V, R, E

class SolveCompressibleBL:
    def __init__(self, Minf, Pr, Tinf, Twall, Gamma, adiabWall):
        self.C2 = 110  # Sutherland Coefficient [Kelvin]
        self.lim = 10
        self.N = 500
        self.h = self.lim / self.N
        self.delta = 1e-10
        self.eps = 1e-9
        self.adi = 1 if adiabWall else 0
        self.Minf = Minf
        self.Pr = Pr
        self.Tinf = Tinf
        self.Twall = Twall
        self.Gamma = Gamma

        self.eta = np.linspace(0, self.lim, self.N + 1)
        self.alfa0 = 0.1
        self.beta0 = 3

        self.y1, self.y2, self.y3, self.y4, self.y5 = self.shoot_method()

    def shoot_method(self):
        y1 = np.zeros(self.N + 1)  # f
        y2 = np.zeros(self.N + 1)  # f'
        y3 = np.zeros(self.N + 1)  # f''
        y4 = np.zeros(self.N + 1)  # rho(eta)
        y5 = np.zeros(self.N + 1)  # rho(eta)'

        for ite in range(100000):
            # Solve using the Runge-Kutta method
            y1, y2, y3, y4, y5 = self.runge_kutta(y1, y2, y3, y4, y5)
            if abs(y2[-1] - 1) < self.eps and abs(y4[-1] - 1) < self.eps:
                break

        return y1, y2, y3, y4, y5

    def runge_kutta(self, y1, y2, y3, y4, y5):
        for i in range(len(self.eta) - 1):
            k11 = y2[i]
            k21 = y3[i]
            k31 = self.y3_rhs(y1[i], y3[i], y4[i], y5[i])
            k41 = y5[i]
            k51 = self.y5_rhs(y1[i], y3[i], y4[i], y5[i])

            k12 = y2[i] + 0.5 * self.h * k21
            k22 = y3[i] + 0.5 * self.h * k31
            k32 = self.y3_rhs(y1[i] + 0.5 * self.h * k11, y3[i] + 0.5 * self.h * k31, y4[i] + 0.5 * self.h * k41, y5[i] + 0.5 * self.h * k51)
            k42 = y5[i] + 0.5 * self.h * k51
            k52 = self.y5_rhs(y1[i] + 0.5 * self.h * k11, y3[i] + 0.5 * self.h * k31, y4[i] + 0.5 * self.h * k41, y5[i] + 0.5 * self.h * k51)

            y5[i + 1] = y5[i] + (self.h / 6) * (k51 + 2 * k52)
            y4[i + 1] = y4[i] + (self.h / 6) * (k41 + 2 * k42)
            y3[i + 1] = y3[i] + (self.h / 6) * (k31 + 2 * k32)
            y2[i + 1] = y2[i] + (self.h / 6) * (k21 + 2 * k22)
            y1[i + 1] = y1[i] + (self.h / 6) * (k11 + 2 * k12)

        return y1, y2, y3, y4, y5

    def y3_rhs(self, y1, y3, y4, y5):
        return -y3 * ((y5 / (2 * y4)) - (y5 / (y4 + self.C2 / self.Tinf))) - y1 * y3 * ((1 / y4) - 1)

    def y5_rhs(self, y1, y3, y4, y5):
        return -2 * y4 * y3 * y1 / (self.Pr * self.Minf**2) - (self.Twall / self.Tinf) * y4 * y5
