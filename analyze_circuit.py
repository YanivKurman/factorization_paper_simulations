##
import pickle
import sinter
import matplotlib.pyplot as plt
import numpy as np
from IR_level_infrastructure import Logical_Experiment
from compiler import compile_ex
from physical_level_infrastructure import ErrorModel
import pickle
import stim
# Generates surface code circuit tasks using Stim's circuit generation.
def generate_tasks(logical_ex: Logical_Experiment, p_vec, d_vec, post_selection, task_name):
    for p in p_vec:
        for d in d_vec:
            circuit=compile_ex(logical_ex, d, ErrorModel(single_qubit_error=0.1 * p, two_qubit_error=p, measurement_error=p)).circ
            filename = f"circuits\{task_name}_d={d}_p={p}.stim"
            with open(filename, "w") as f:
                f.write(str(circuit))
            model = circuit.detector_error_model(decompose_errors=True)
            mask=sinter.post_selection_mask_from_4th_coord(model) # need to verify that this makes sense maybe the mask is just wrong
            if post_selection:
                yield sinter.Task(
                    circuit=circuit,
                    json_metadata={
                        'p': p,
                        'd': d,
                    },
                    postselection_mask=mask,
                )
            else:
                yield sinter.Task(
                    circuit=circuit,
                    json_metadata={
                        'p': p,
                        'd': d,
                    },
                )
def generate_tasks_from_circuits(p_vec, d_vec, post_selection, task_name):
    for p in p_vec:
        for d in d_vec:
            filename = f"circuits\{task_name}_d={d}_p={p}.stim"
            with open(filename, "r") as f:
                circuit = stim.Circuit(f.read())
            model = circuit.detector_error_model(decompose_errors=True)
            mask = sinter.post_selection_mask_from_4th_coord(
                model)
            print(d)
            if post_selection:
                yield sinter.Task(
                    circuit=circuit,
                    json_metadata={
                        'p': p,
                        'd': d,
                    },
                    postselection_mask=mask,
                )
            else:
                yield sinter.Task(
                    circuit=circuit,
                    json_metadata={
                        'p': p,
                        'd': d,
                    },
                )


def analyze(logical_ex: Logical_Experiment, p_vec, d_vec, post_selection:bool, task_name: str= ' '):
    samples = sinter.collect(
        num_workers=5,
        max_shots=20_000_000,
        max_errors=500,
        tasks=generate_tasks(logical_ex, p_vec, d_vec, post_selection, task_name=task_name),
        decoders=['pymatching'],
    )

    # Print samples as CSV data.
    print(sinter.CSV_HEADER)
    for sample in samples:
        print(sample.to_csv_line())
    # save results to file
    with open(task_name+'.pkl', 'wb') as f:
        pickle.dump(samples, f)

##
def analyze_from_circ(p_vec, d_vec, post_selection:bool, task_name: str= ' '):
    samples = sinter.collect(
        num_workers=4,
        max_shots=30_000_000,
        max_errors=1200,
        tasks=generate_tasks_from_circuits(p_vec, d_vec, post_selection, task_name),
        decoders=['pymatching'],
    )

    # Print samples as CSV data.
    print(sinter.CSV_HEADER)
    for sample in samples:
        print(sample.to_csv_line())
    minimal_error=1
    for sample in samples:
        if sample.shots-sample.discards:
            error=sample.errors/(sample.shots-sample.discards)
            if error<minimal_error:
                minimal_error=error

    # Render a matplotlib plot of the data.
    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=samples,
        group_func=lambda stat: f" d={stat.json_metadata['d']}",
        x_func=lambda stat: stat.json_metadata['p'],
    )
    ax.loglog()
    ax.set_ylim(10 ** np.floor(np.log10(10 ** np.floor(np.log10(minimal_error)))), 1)
    ax.set_ylim(minimal_error*0.5, 1)
    ax.grid()
    ax.set_title(f"{task_name}_PS={post_selection}")
    ax.set_ylabel('Logical Error Probability (per shot)')
    ax.set_xlabel('Physical Error Rate')
    ax.legend()
    # Save figure to file and also open in a window.
    fig.savefig(f"{task_name}_PS={post_selection}.png")
    plt.show()
    # save results to file
    with open(f"{task_name}_PS={post_selection}.pkl", 'wb') as f:
        pickle.dump(samples, f)

    # with open(task_name+'.pkl', 'rb') as f: #load results
    #     loaded_tasks_stats = pickle.load(f)

##
from physical_level_infrastructure import Experiment
from IR_level_infrastructure import Logical_Experiment
def get_num_measurements(ex: Experiment):
    return ex.circ.num_measurements

def get_num_logical_measurements(ex: Experiment):
    return len(ex.logical_measurements)

def get_num_qubits(ex: Experiment):
    return len(ex.physical_qubits)

def get_num_detectors(ex: Experiment):
    return ex.circ.num_detectors

def total_stabilizer_rounds(ex: Experiment,logical_ex:Logical_Experiment):
    tick_count = 0
    for ops in logical_ex.circ:
        if ops.name == 'TICK':
            tick_count += 1
    d = len(ex.logical_measurements[-1][0])
    return d * (tick_count+1)


def get_avg_data_creation_rate(ex: Experiment, logical_ex: Logical_Experiment):
    return get_num_measurements(ex) / total_stabilizer_rounds(ex, logical_ex)

def get_avg_alive_surface(logical_ex: Logical_Experiment):
    blocks=get_nFT_d3_blocks(logical_ex)+get_FT_d3_blocks(logical_ex)
    end_tick = max([index for index, element in enumerate(logical_ex.activated_qubits) if element != []])+1
    return blocks/end_tick

def get_max_parallel_CNOTS(ex: Experiment,logical_ex:Logical_Experiment):
    d=len(ex.logical_measurements[-1][0])
    max_surfaces=get_max_activated_surfaces(logical_ex)
    return max_surfaces*(d*(d-1))
def get_max_parallel_measurements(ex: Experiment,logical_ex:Logical_Experiment):
    largest_meas=0
    lists=logical_ex.activated_qubits
    # Iterate through the list of lists, except the last element
    for i in range(len(lists) - 1):
        # Calculate the absolute difference in length between neighboring lists
        meas = len(lists[i]) - len(lists[i+1])+len(lists[i])
        # Update the largest difference if the current difference is greater
        if meas > largest_meas:
            largest_meas = meas
            largest_tick=i
    d = len(ex.logical_measurements[-1][0])
    return d**2*(len(lists[largest_tick]) - len(lists[largest_tick+1]))+len(lists[largest_tick])*(d**2-1)

def get_max_activated_surfaces(logical_ex: Logical_Experiment):
    return max([len(sublist) for sublist in logical_ex.activated_qubits])
def get_frame_propagations(logical_ex: Logical_Experiment):
    return len(logical_ex.frame_propagation)

def get_nFT_d3_blocks(logical_ex: Logical_Experiment):
    count=0
    for inst in logical_ex.circ:
        if inst.name in {'S','S_DAG'}:
            count+=len(inst.targets_copy())
    return count

def get_FT_d3_blocks(logical_ex: Logical_Experiment):
    surfaces=logical_ex.activated_qubits
    blocks=len(surfaces[0])
    for i in range(1,len(surfaces) - 1):
        set1 = set(surfaces[i-1])
        set2 = set(surfaces[i])
        blocks+=len(surfaces[i])+len(set1.difference(set2))
    return blocks-get_nFT_d3_blocks(logical_ex)



def get_resources(ex: Experiment, logical_ex: Logical_Experiment): #need to verify!!
    ex_resources = {
        'num_measurements': get_num_measurements(ex),
        'num_physical_qubits': get_num_qubits(ex),
        'num_graph_nodes': get_num_detectors(ex),
        'num_logical_measurements': get_num_logical_measurements(ex),
        'total_rounds': total_stabilizer_rounds(ex, logical_ex),
        'avg_data_creation_rate [bit/round]': get_avg_data_creation_rate(ex, logical_ex),
        'avg_alive_surface': get_avg_alive_surface(ex, logical_ex),
        'max_parallel_cnots': get_max_parallel_CNOTS(ex, logical_ex),  # this does not include surgery CNOTS
        'max_parallel_measurements': get_max_parallel_measurements(ex, logical_ex),  # this does not include surgery measurements
        'num_frame_propagation_gates': get_frame_propagations(logical_ex),
        'num_FT_d3_blocks': get_FT_d3_blocks(logical_ex),
        'num_nFT_d3_blocks': get_nFT_d3_blocks(logical_ex),
    }
    return ex_resources

## cxreates interactive circuit
import webbrowser
def create_interactive_html(exp: Experiment,task_name: str= ' '):
    html_content = exp.circ.diagram('interactive-html')._repr_html_()
    file_name = task_name+'interactive.html'
    with open(file_name, 'w', encoding='utf-8') as html_file:
        html_file.write(html_content)
    webbrowser.open(file_name)
## count gates
import stim
filename = "circuits/fact_circuit_d=5_p=0.001.stim"
with open(filename, "r") as f:
    circuit = stim.Circuit(f.read())
counter = 0
for i in range(len(circuit)):
    if circuit[i].name in['R','RX','H','M','SQRT_X']:
        counter+=len(circuit[i].targets_copy())
    if circuit[i].name in ['CZ', 'CX']:
        counter += len(circuit[i].targets_copy())/2
print(counter)