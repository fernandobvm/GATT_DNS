import numpy as np
from getDomainSlices import get_domain_slices
class MatrixTypeBlocks:
    def __init__(self, type_map, p_row, p_col, max_block_size=128):
        self.type_map = type_map
        self.p_row = p_row
        self.p_col = p_col
        self.max_block_size = max_block_size
        
        self.blocks = self.get_matrix_type_blocks()

    def get_matrix_type_blocks(self):
        J, K = self.type_map.shape
        all_blocks = []

        # Get all blocks
        for k in range(K):
            starts = np.concatenate(([0], np.where(np.diff(self.type_map[:, k]) != 0)[0] + 1))
            ends = np.concatenate((starts[1:] - 1, [J - 1]))
            for start, end in zip(starts, ends):
                all_blocks.append([self.type_map[start, k], start, end, k, start, end])

        all_blocks = np.array(all_blocks)

        # Merge blocks
        i = 0
        while i < len(all_blocks):
            j = i + 1
            while j < len(all_blocks):
                if np.array_equal(all_blocks[i, :3], all_blocks[j, :3]) and all_blocks[i, 4] + 1 == all_blocks[j, 3]:
                    all_blocks[i, 5] = all_blocks[j, 5]
                    all_blocks = np.delete(all_blocks, j, axis=0)
                else:
                    j += 1
            i += 1

        # Divide blocks for processors
        domain_slices_y = self.get_domain_slices(J, self.p_row)
        domain_slices_z = self.get_domain_slices(K, self.p_col)

        blocks = [None] * (self.p_row * self.p_col)

        for j in range(self.p_row):
            for k in range(self.p_col):
                n_proc = k + j * self.p_col
                bL = all_blocks.copy()

                Ji, Jf = domain_slices_y[j]
                Ki, Kf = domain_slices_z[k]

                bL = bL[(bL[:, 1] <= Jf) & (bL[:, 2] >= Ji) & (bL[:, 3] <= Kf) & (bL[:, 4] >= Ki)]

                bL[:, 1] = np.maximum(bL[:, 1], Ji)
                bL[:, 2] = np.minimum(bL[:, 2], Jf)
                bL[:, 3] = np.maximum(bL[:, 3], Ki)
                bL[:, 4] = np.minimum(bL[:, 4], Kf)

                blocks[n_proc] = bL

        # Reduce block sizes if needed
        if not np.isinf(self.max_block_size):
            for n_proc in range(self.p_row * self.p_col):
                bL = blocks[n_proc]
                bL_new = []

                for row in bL:
                    i_size = row[2] - row[1] + 1
                    j_size = row[4] - row[3] + 1
                    b_size = i_size * j_size
                    n_slices = int(np.ceil(b_size / self.max_block_size))

                    if n_slices == 1:
                        bL_new.append(row)
                    else:
                        i_slices = int(np.ceil(n_slices / j_size))
                        j_slices = n_slices // i_slices

                        i_slices_ind = self.get_domain_slices(i_size, i_slices)
                        j_slices_ind = self.get_domain_slices(j_size, j_slices)

                        for ii in range(i_slices):
                            for jj in range(j_slices):
                                bL_new.append([row[0], 
                                               i_slices_ind[ii][0] + row[1] - 1, 
                                               i_slices_ind[ii][1] + row[1] - 1, 
                                               j_slices_ind[jj][0] + row[3] - 1, 
                                               j_slices_ind[jj][1] + row[3] - 1])

                blocks[n_proc] = np.array(bL_new)

        return blocks


# Uso da classe
# type_map = np.array(...)  # Suponha que este array esteja definido
# matrix_blocks = MatrixTypeBlocks(type_map, p_row, p_col)
