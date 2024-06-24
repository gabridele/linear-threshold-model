import numpy as np
import pandas as pd # type: ignore
import sys

def restore_matrix(matrix_ass, matrix_zero):
    
    original_size = 454
    if matrix_ass.shape == (original_size, original_size):
        return matrix_ass
    else:
        restored_matrix = np.zeros((original_size, original_size))
        rows_present = np.any(matrix_zero, axis=1)
        cols_present = np.any(matrix_zero, axis=0)
        row_indices = np.where(rows_present)[0]
        col_indices = np.where(cols_present)[0]

        for i, row in enumerate(row_indices):
            for j, col in enumerate(col_indices):
                restored_matrix[row, col] = matrix_ass[i, j]

        return restored_matrix

def main(input_ass, input_zero, n_seeds):
    sub_id = input_ass.split('/')[-3]
    
    print('subject:', sub_id)
    
    matrix_ass = pd.read_csv(input_ass, delimiter=',', header=None).to_numpy().astype(float)
    matrix_zero = pd.read_csv(input_zero, delimiter=',', header=None).to_numpy().astype(float)
    
    restored_matrix = restore_matrix(matrix_ass, matrix_zero)

    restored_matrix_filename = f"derivatives/{sub_id}/dwi/full_association_mtrix_{sub_id}_{n_seeds}seeds.csv"

    np.savetxt(restored_matrix_filename, restored_matrix, delimiter=",")

if __name__ == "__main__":

    input_ass = sys.argv[1]
    input_zero = sys.argv[2]
    n_seeds = sys.argv[3]
    
    main(input_ass, input_zero, n_seeds)
