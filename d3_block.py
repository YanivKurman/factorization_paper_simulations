from logical_level_circuit import build_memory
from analyze_circuit import analyze
import pickle
import numpy as np

run_simulation=False
save_matrices=True
p_vec = [0.001, 0.003, 0.005, 0.008, 0.01]
d_vec = [3, 5, 7, 9]

if run_simulation:
    logical_ex=build_memory()
    analyze(logical_ex, p_vec, d_vec, post_selection=False, task_name='Memory')
else: #load

    file_path = 'results/Memory.pkl'
    # Open the file in binary read mode
    with open(file_path, 'rb') as file:
        # Load the object from the file
        data = pickle.load(file)

    d3_errors=np.empty((len(p_vec), len(d_vec)))
    for indx in range(len(data)):
        ind_p = p_vec.index(data[indx].json_metadata['p'])
        ind_d = d_vec.index(data[indx].json_metadata['d'])
        d3_errors[ind_p, ind_d] = data[indx].errors / (data[indx].shots - data[indx].discards)

    if save_matrices:
        np.save('matrices/d3_errors_matrix.npy', d3_errors)
        # np.save('matrices/d3_p_vec.npy', p_vec)
        # np.save('matrices/d3_d_vec.npy', d_vec)

# # Load the matrix from the file
# loaded_matrix = np.load('d3_errors_matrix.npy')
#
# # Display the loaded matrix
# print(loaded_matrix)
