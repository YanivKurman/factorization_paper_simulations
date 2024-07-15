import numpy as np
from logical_level_circuit import build_memory_18d3
import pickle
import matplotlib.pyplot as plt
import numpy as np
from IR_level_infrastructure import Logical_Experiment
from compiler import compile_ex
from physical_level_infrastructure import ErrorModel

task_size=np.array([11,12,39,15,15,28,30,5,16,53,75,2.5,4.5])
avg_size=task_size.mean()
median=np.median(task_size)

p_vec=[0.001,0.003,0.005,0.007]
d_vec=[3,5,7,9]

logical_ex=build_memory_18d3()
d3_syndromes=np.empty((len(p_vec), len(d_vec)))

for ind_p,p in enumerate(p_vec):
    for ind_d,d in enumerate(d_vec):
        circuit = compile_ex(logical_ex, d,
                             ErrorModel(single_qubit_error=0.1 * p, two_qubit_error=p, measurement_error=p)).circ
        model = circuit.detector_error_model(decompose_errors=True)
        num_shots = 10000
        sampler = circuit.compile_detector_sampler()
        syndrome, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)
        d3_syndromes[ind_p,ind_d]=sum(sum(syndrome))/num_shots/18



## saving results
import numpy as np

np.save('avg_syndromes_d3_matrix.npy', d3_syndromes)
np.save('avg_syndrome_p_vec.npy', p_vec)
np.save('avg_syndrome_d_vec.npy', d_vec)


##
expanded_tasks = task_size[:, np.newaxis, np.newaxis]
product_matrix = expanded_tasks * d3_syndromes
##
plt.figure(figsize=(10, 6))

# Define colors for each d value
colors = ['blue', 'green', 'red', 'purple']
shifts = np.linspace(-0.0002, 0.0002, len(d_vec))  # Small shifts for each d to avoid overlap

# Scatter plot for each d, with slight shifts to avoid overlap
for i, d in enumerate(d_vec):
    for j, p in enumerate(p_vec):
        # Get all points for this d and p
        data = product_matrix[:, i, j]/d
        shifted_p = p + shifts[i]  # Slight shift for each d
        plt.scatter([shifted_p] * len(data), data, color=colors[i],alpha=0.5, label=f'd={d}' if j == 0 else "")

        # Calculate and plot the average for each group
        # if j == len(p_vec) - 1:  # Only add average marker at the last p to avoid duplicate entries in legend
        #     average = np.mean(data)
        #     plt.scatter(shifted_p, average, color=colors[i], marker='D', s=50, label=f'Avg for d={d}' if j == 0 else "")

# Adding labels and title
plt.title('Scatter Plot of All Points in Product Matrix with Shifts and Averages')
plt.xlabel('p_vec Values')
plt.ylabel('Product Values')
plt.yscale('log')
plt.legend()

# Show the plot
plt.show()

##
import matplotlib.pyplot as plt
import numpy as np

# Choose which d to use for the bottom x-axis (e.g., d=1 for the second row in d3_syndromes)
d_index = 1  # Second row, change as needed

# Create a figure and primary axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar plot on the primary y-axis
bars = ax1.barh(range(len(task_size)), task_size, color='lightblue')
ax1.set_xlabel('Task Size')
ax1.set_yticks(range(len(task_size)))
ax1.set_yticklabels([f'Task {i+1}' for i in range(len(task_size))])

# Secondary x-axis (top)
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())  # Ensure the limits of secondary axis match the primary axis
ax2.set_xticks(ax1.get_xticks())  # Align ticks with the primary x-axis

# Show the plot
plt.show()

##
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(20, 15))
axs = axs.flatten()  # Flatten the array of axes for easier indexing

# First plot: task_size
# axs[0].barh(range(len(task_size)), task_size, color='navy')
# axs[0].set_xlabel('Task Size')
# axs[0].set_yticks(range(len(task_size)))
# # axs[0].set_yticklabels([f'Task {i+1}' for i in range(len(task_size))])

# Next plots: each combination of d and p
for i in range(4):  # d dimension
    for j in range(4):  # p dimension
        index = i * 4 + j  # Calculate subplot index
        ax = axs[index]
        ax.barh(range(len(task_size)), product_matrix[:, i, j], color=plt.cm.viridis(j / 3))
        ax.set_title(f'd={d_vec[i]}, p={p_vec[j]:.3f}')
        ax.set_yticks(range(len(task_size)))
        # ax.set_xticklabels([f'Task {k+1}' for k in range(len(task_size))], rotation=90)

# Adjust layout
# plt.tight_layout()
plt.show()
## latency punishement per task
import numpy as np
import matplotlib.pyplot as plt

# Load data
active_logic_surfaces = [3, 3, 3, 3, 5, 5, 5, 5, 4, 2, 1, 1]
d3_mat = np.load('d3_errors_matrix.npy')
d_vec = np.load('d3_d_vec.npy')
p_vec = np.load('d3_p_vec.npy')
maximal_logical_surfaces = 5
latency = np.arange(50) + 1

# Define line styles and colors
line_styles = ['-', '--']
colors = plt.cm.viridis(np.linspace(0, 1, len(d_vec)))

plt.figure(figsize=(10, 6))
for p_ind in [0, 1]:
    for d_ind in range(len(d_vec)):
        added_error = maximal_logical_surfaces * d3_mat[p_ind, d_ind] * latency / d_vec[d_ind]
        plt.plot(latency, added_error, linestyle=line_styles[p_ind], color=colors[d_ind], label=f'd={d_vec[d_ind]}, p={p_ind}')

plt.xlabel('latency delay/T_round')
plt.ylabel('additional error')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()
