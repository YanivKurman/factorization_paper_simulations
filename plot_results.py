import pickle
import numpy as np
import sinter
import matplotlib.pyplot as plt

print_results=True
plot_results_error_vs_d=True
plot_results_error_vs_p=True

file_path = 'results/Memory.pkl' # Replace 'your_file.pkl' with the path to your .pkl file
with open(file_path, 'rb') as file:
    # Load the object from the file
    data = pickle.load(file)


if print_results:
    print(sinter.CSV_HEADER)
    for sample in data:
        print(sample.to_csv_line())


if plot_results_error_vs_d:
    # Render a matplotlib plot of the data.
    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=data,
        group_func=lambda stat: f" p={stat.json_metadata['p']}",
        x_func=lambda stat: stat.json_metadata['d'],
    )
    ax.set_ylim(0, 1)
    ax.grid()
    plt.xticks([3, 5, 7, 9])
    plt.yticks(np.arange(0, 1.1, 0.1))

    ax.set_ylabel('Logical Error Probability (per shot)')
    ax.set_xlabel('distance')
    ax.legend()
    # Save figure to file and also open in a window.
    plt.show()

if plot_results_error_vs_p:
    # Render a matplotlib plot of the data.
    minimal_error = 1
    for sample in data:
        if sample.shots - sample.discards:
            error = sample.errors / (sample.shots - sample.discards)
            if error < minimal_error:
                minimal_error = error

    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=data,
        group_func=lambda stat: f" d={stat.json_metadata['d']}",
        x_func=lambda stat: stat.json_metadata['p'],
    )
    ax.loglog()

    ax.set_ylim(10 ** np.floor(np.log10(10 ** np.floor(np.log10(minimal_error)))), 1)
    ax.set_ylim(minimal_error*0.5, 1)
    ax.grid()
    ax.set_ylabel('Logical Error Probability (per shot)')
    ax.set_xlabel('Physical Error Rate')
    ax.legend()
    plt.show()

