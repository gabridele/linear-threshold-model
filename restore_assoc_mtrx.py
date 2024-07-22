import numpy as np
import pandas as pd # type: ignore
import sys

def zero_out_low_connections(matrix, threshold=5):
    """
    Zero out nodes (rows and columns) in the matrix that have threshold or fewer connections.
    """
    # Count non-zero entries in each row and column, which correspond to node connections
    row_nonzero_counts = np.count_nonzero(matrix, axis=1)
    col_nonzero_counts = np.count_nonzero(matrix, axis=0)

    # Identify rows and columns to zero out based on the threshold
    nodes_to_zero = (row_nonzero_counts <= threshold) | (col_nonzero_counts <= threshold)

    # Zero out the identified rows and columns
    matrix[nodes_to_zero, :] = 0
    matrix[:, nodes_to_zero] = 0

    return matrix

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
    
    # Restore matrix (adds back zero-entry nodes)
    restored_matrix = restore_matrix(matrix_ass, matrix_zero)
    
    # Zero out nodes with 5 or fewer connections
    restored_matrix = zero_out_low_connections(restored_matrix, threshold=5)
    
    # Save modified matrix
    restored_matrix_filename = f"derivatives/{sub_id}/dwi/restored_full_association_matrix_{sub_id}_{n_seeds}seeds.csv"
    np.savetxt(restored_matrix_filename, restored_matrix, delimiter=",")

if __name__ == "__main__":

    input_ass = sys.argv[1]
    input_zero = sys.argv[2]
    n_seeds = sys.argv[3]
    
    main(input_ass, input_zero, n_seeds)
