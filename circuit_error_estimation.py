import pickle
import numpy as np
import sinter
import matplotlib.pyplot as plt
import scipy.special as sp

print_results=True
plot=True
plot_results_error_vs_p=True

file_path = 'results/S_gate_PS=True.pkl' # Replace 'your_file.pkl' with the path to your .pkl file
with open(file_path, 'rb') as file:
    # Load the object from the file
    data_S = pickle.load(file)


def effective_error_probability(p, n):
    prob = 0
    for k in range(1, n+1, 2):
        prob += sp.comb(n, k) * (p**k) * ((1-p)**(n-k))
    return prob


def combined_error_probability(p1, n1, p2, n2):
    P_eff_1 = effective_error_probability(p1, n1)
    P_eff_2 = effective_error_probability(p2, n2)

    # Combined probability of odd number of errors
    P_combined = P_eff_1 + P_eff_2 - 2 * P_eff_1 * P_eff_2
    return P_combined

##
p_vec=[0.001,0.003,0.005,0.008,0.01]
d_vec=[3,5,7,9]

N_nFT=14
N_FT=296
d3_mat = np.load('matrices/d3_errors_matrix.npy')
nFT_mat = 1-np.load('matrices/nFT_PS_init.npy')
total_mat=np.zeros([len(p_vec),len(d_vec)])
for ind_p in range(len(p_vec)):
    for ind_d in range(len(d_vec)):
        total_mat[ind_p,ind_d]=combined_error_probability(d3_mat[ind_p,ind_d], N_FT, nFT_mat[ind_p,ind_d], N_nFT)
##
import matplotlib.pyplot as plt

# Create a figure and subplots
fig, axs = plt.subplots(3, 2, figsize=(14, 12))

# Plot d3_mat
# d3_mat with x-axis as d
for i, row in enumerate(d3_mat):
    axs[0, 0].plot(d_vec, row, marker='o', label=f'p={p_vec[i]}')
axs[0, 0].set_title('d3_mat (x-axis: d)')
axs[0, 0].set_xlabel('d')
axs[0, 0].set_ylabel('Value')
axs[0, 0].legend()
axs[0, 0].set_yscale('log')  # Add log scale to y-axis

# d3_mat with x-axis as p
for i, col in enumerate(d3_mat.T):
    axs[0, 1].plot(p_vec, col, marker='o', label=f'd={d_vec[i]}')
axs[0, 1].set_title('d3_mat (x-axis: p)')
axs[0, 1].set_xlabel('p')
axs[0, 1].set_ylabel('Value')
axs[0, 1].legend()
axs[0, 1].set_yscale('log')  # Add log scale to y-axis
axs[0, 1].set_xscale('log')  # Add log scale to y-axis
# Plot nFT_mat
# nFT_mat with x-axis as d
for i, row in enumerate(nFT_mat):
    axs[1, 0].plot(d_vec, row, marker='o', label=f'p={p_vec[i]}')
axs[1, 0].set_title('nFT_mat (x-axis: d)')
axs[1, 0].set_xlabel('d')
axs[1, 0].set_ylabel('Value')
axs[1, 0].legend()
axs[1, 0].set_yscale('log')  # Add log scale to y-axis

# nFT_mat with x-axis as p
for i, col in enumerate(nFT_mat.T):
    axs[1, 1].plot(p_vec, col, marker='o', label=f'd={d_vec[i]}')
axs[1, 1].set_title('nFT_mat (x-axis: p)')
axs[1, 1].set_xlabel('p')
axs[1, 1].set_ylabel('Value')
axs[1, 1].legend()
axs[1, 1].set_yscale('log')  # Add log scale to y-axis
axs[1, 1].set_xscale('log')  # Add log scale to y-axis

# Plot total_mat
# total_mat with x-axis as d
for i, row in enumerate(total_mat):
    axs[2, 0].plot(d_vec, row, marker='o', label=f'p={p_vec[i]}')
axs[2, 0].set_title('total_mat (x-axis: d)')
axs[2, 0].set_xlabel('d')
axs[2, 0].set_ylabel('Value')
axs[2, 0].legend()
axs[2, 0].set_yscale('log')  # Add log scale to y-axis

# total_mat with x-axis as p
for i, col in enumerate(total_mat.T):
    axs[2, 1].plot(p_vec, col, marker='o', label=f'd={d_vec[i]}')
axs[2, 1].set_title('total_mat (x-axis: p)')
axs[2, 1].set_xlabel('p')
axs[2, 1].set_ylabel('Value')
axs[2, 1].legend()
axs[2, 1].set_yscale('log')  # Add log scale to y-axis
axs[2, 1].set_xscale('log')  # Add log scale to y-axis

# Adjust layout
plt.tight_layout()
plt.show()
