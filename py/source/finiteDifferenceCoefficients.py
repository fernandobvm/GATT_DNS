import numpy as np

class FiniteDifferenceCoefficients:
    def __init__(self, method):
        self.method = method
        self.centered_stencil_lhs, self.centered_stencil_rhs, self.decentered_stencil_lhs, self.decentered_stencil_rhs = self.calculate_coefficients()

    def calculate_coefficients(self):
        centered_stencil_lhs = None
        centered_stencil_rhs = None
        decentered_stencil_lhs = None
        decentered_stencil_rhs = None

        if self.method == 'SL6':
            w = 1.8

            # Center
            matriz_a = np.array([[1, 1, 1, -2],
                                  [1, 4, 9, -6],
                                  [1, 16, 81, -10],
                                  [np.sin(w), np.sin(2 * w) / 2, np.sin(3 * w) / 3, -2 * w * np.cos(w)]])
            matriz_b = np.array([1, 0, 0, w])

            coeffs = np.linalg.solve(matriz_a, matriz_b)

            a = coeffs[0] / 2
            b = coeffs[1] / 4
            c = coeffs[2] / 6
            alpha = coeffs[3]

            centered_stencil_lhs = np.array([1, alpha])
            centered_stencil_rhs = np.array([0, a, b, c])

            # Border
            # First stage
            matriz_a = np.array([[1, 1, 1, 1, 1, 1, 0],
                                  [0, 1, 2, 3, 4, 5, -1],
                                  [0, 1, 4, 9, 16, 25, -2],
                                  [0, 1, 8, 27, 64, 125, -3],
                                  [0, 1, 16, 81, 256, 625, -4],
                                  [0, 1, 32, 243, 1024, 3125, -5],
                                  [0, 1, 64, 729, 4096, 15625, -6]])
            matriz_b = np.array([0, 1, 0, 0, 0, 0, 0])

            coeffs = np.linalg.solve(matriz_a, matriz_b)

            a = coeffs[0]
            b = coeffs[1]
            c = coeffs[2]
            d = coeffs[3]
            e = coeffs[4]
            f = coeffs[5]
            alpha = coeffs[6]

            matriz_lhs_aux = np.array([1, alpha, 0, 0])
            matriz_rhs_aux = np.array([a, b, c, d, e, f])

            # Second stage
            matriz_a = np.array([[-1, 2],
                                  [-1, 6]])

            matriz_b = np.array([-1, 0])

            coeffs = np.linalg.solve(matriz_a, matriz_b)

            a = coeffs[0] / 2
            alpha = coeffs[1]

            matriz_lhs_aux2 = np.array([alpha, 1, alpha, 0])
            matriz_rhs_aux2 = np.array([-a, 0, a, 0, 0, 0])

            # Third stage (same as centered SL4)
            matriz_a = np.array([[1, 0, -2/3],
                                  [0, 1, -4/3],
                                  [np.sin(w), np.sin(2 * w) / 2, -2 * w * np.cos(w)]])

            matriz_b = np.array([4/3, -1/3, w])

            coeffs = np.linalg.solve(matriz_a, matriz_b)

            a = coeffs[0] / 2
            b = coeffs[1] / 4
            alpha = coeffs[2]

            matriz_lhs_aux3 = np.array([0, alpha, 1, alpha])
            matriz_rhs_aux3 = np.array([-b, -a, 0, a, b, 0])

            decentered_stencil_lhs = np.vstack([matriz_lhs_aux,
                                                 matriz_lhs_aux2,
                                                 matriz_lhs_aux3])

            decentered_stencil_rhs = np.vstack([matriz_rhs_aux,
                                                 matriz_rhs_aux2,
                                                 matriz_rhs_aux3])

        elif self.method == 'SL6O3':
            coeffs = [0.392465753424658, 1.565410958904110, 0.237260273972603, -0.017739726027397]
            alpha, a, b, c = coeffs
            centered_stencil_lhs = np.array([1, alpha])
            centered_stencil_rhs = np.array([0, a / 2, b / 4, c / 6])

            coeffs_P3 = [0.350978473581213, 1.567318982387476, 0.134637964774951]
            alpha_P3, a_P3, b_P3 = coeffs_P3

            decentered_stencil_lhs = np.array([[1, (3 * np.pi + 40) / (3 * np.pi + 8), 0, 0],
                                                [1 / 4, 1, 1 / 4, 0],
                                                [0, alpha_P3, 1, alpha_P3]])
            decentered_stencil_rhs = np.array([[-(13 * np.pi + 56) / (2 * (3 * np.pi + 8)), 
                                                 (15 * np.pi + 8) / (2 * (3 * np.pi + 8)), 
                                                 -(3 * np.pi - 56) / (2 * (3 * np.pi + 8)), 
                                                 (np.pi - 8) / (2 * (3 * np.pi + 8)), 0],
                                                [-3 / 4, 0, 3 / 4, 0, 0],
                                                [-b_P3 / 4, -a_P3 / 2, 0, a_P3 / 2, b_P3 / 4]])

        elif self.method == 'SL4':
            w = 1.8
            matriz_a = np.array([[1, 0, -2/3],
                                  [0, 1, -4/3],
                                  [np.sin(w), np.sin(2 * w) / 2, -2 * w * np.cos(w)]])

            matriz_b = np.array([4/3, -1/3, w])

            coeffs = np.linalg.solve(matriz_a, matriz_b)

            a = coeffs[0] / 2
            b = coeffs[1] / 4
            alpha = coeffs[2]

            centered_stencil_lhs = np.array([1, alpha])
            centered_stencil_rhs = np.array([0, a, b])

            decentered_stencil_lhs = np.array([[1, 3, 0],
                                                [1/4, 1, 1/4]])

            decentered_stencil_rhs = np.array([[-17/6, 3/2, 3/2, -1/6],
                                                [-3/4, 0, 3/4, 0]])

        elif self.method == 'EX2':
            centered_stencil_lhs = 1
            centered_stencil_rhs = np.array([0, 1/2])
            decentered_stencil_lhs = 1
            decentered_stencil_rhs = np.array([-3/2, 2, -1/2])

        elif self.method == 'EX4':
            centered_stencil_lhs = 1
            centered_stencil_rhs = np.array([0, 2/3, -1/12])
            decentered_stencil_lhs = 1
            decentered_stencil_rhs = np.array([[-25/12, 4, -3, 4/3, -1/4],
                                                [-1/2, 0, 1/2, 0, 0]])

        else:
            raise ValueError(f'Finite differences method not implemented: {self.method}. Check finiteDifferenceCoefficients for available methods')

        return centered_stencil_lhs, centered_stencil_rhs, decentered_stencil_lhs, decentered_stencil_rhs
