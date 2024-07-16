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
        max_shots=30_000_000,
        max_errors=1200,
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

## resource extraction

import stim
from logical_level_circuit import build_factorization_circuit


def get_num_measurements(circ: stim.Circuit):
    return circ.num_measurements


def get_num_qubits(circ: stim.Circuit):
    qubits = set()
    for operation in circ:
        if operation.name in ['R', 'RX']:  # Assuming 'R' is the restart operation
            for target in operation.targets_copy():
                if target.is_qubit_target:
                    qubits.add(target.value)
    return len(qubits)


def get_num_detectors(circ: stim.Circuit):
    return circ.num_detectors


def get_max_active_qubits(circuit: stim.Circuit) -> int:
    active_qubits = set()
    max_active = 0

    for operation in circuit:
        if operation.name == 'R':  # Assuming 'R' is the restart operation (initialization)
            for target in operation.targets_copy():
                if target.is_qubit_target:
                    active_qubits.add(target.value)
        elif operation.name == 'M':  # Assuming 'M' is the measurement operation
            for target in operation.targets_copy():
                if target.is_qubit_target:
                    active_qubits.discard(target.value)

        # Update the maximum number of active qubits
        max_active = max(max_active, len(active_qubits))

    return max_active


def get_max_parallel_measurements(circuit: stim.Circuit) -> int:
    current_measurements = 0
    max_measurements = 0

    for operation in circuit:
        if operation.name == 'M':  # Assuming 'M' is the measurement operation
            current_measurements += len([target for target in operation.targets_copy() if target.is_qubit_target])
        elif operation.name == 'TICK':
            # Update the maximum count and reset the current count
            max_measurements = max(max_measurements, current_measurements)
            current_measurements = 0

    # Final check in case the circuit does not end with a 'TICK'
    max_measurements = max(max_measurements, current_measurements)

    return max_measurements


def max_two_qubit_gates(circuit: stim.Circuit) -> int:
    current_two_qubit_gates = 0
    max_two_qubit_gates = 0

    for operation in circuit:
        if operation.name in {'CX', 'CZ'}:  # Check for two-qubit gates
            current_two_qubit_gates += len(
                operation.targets_copy()) // 2  # Each target pair represents a two-qubit gate
        elif operation.name == 'TICK':
            # Update the maximum count and reset the current count
            max_two_qubit_gates = max(max_two_qubit_gates, current_two_qubit_gates)
            current_two_qubit_gates = 0

    # Final check in case the circuit does not end with a 'TICK'
    max_two_qubit_gates = max(max_two_qubit_gates, current_two_qubit_gates)

    return max_two_qubit_gates


def total_stabilizer_rounds(logical_ex: Logical_Experiment, d):
    tick_count = 0
    for ops in logical_ex.circ:
        if ops.name == 'TICK':
            tick_count += 1
    return d * (tick_count + 1)


def get_avg_data_creation(circ: stim.Circuit, logical_ex: Logical_Experiment, d) -> int:
    return get_num_measurements(circ) / total_stabilizer_rounds(logical_ex, d)


def get_nFT_d3_blocks(logical_ex: Logical_Experiment):
    count = 0
    for inst in logical_ex.circ:
        if inst.name in {'S', 'S_DAG'}:
            count += len(inst.targets_copy())
    return count


def get_FT_d3_blocks(logical_ex: Logical_Experiment):
    surfaces = logical_ex.activated_qubits
    blocks = len(surfaces[0])
    for i in range(1, len(surfaces) - 1):
        set1 = set(surfaces[i - 1])
        set2 = set(surfaces[i])
        blocks += len(surfaces[i]) + len(set1.difference(set2))
    return blocks - get_nFT_d3_blocks(logical_ex)


def get_resources(task_name, d):
    filename = f"circuits\{task_name}_d={d}_p=0.001.stim"
    with open(filename, "r") as f:
        circ = stim.Circuit(f.read())
    logical_ex = build_factorization_circuit()
    ex_resources = {
        'num_physical_qubits': get_num_qubits(circ),
        'max_active_qubits': get_max_active_qubits(circ),
        'max_parallel_measurements': get_max_parallel_measurements(circ),
        'max_parallel_cnots': max_two_qubit_gates(circ),  # this does not include surgery CNOTS
        'num_measurements': get_num_measurements(circ),
        'num_graph_nodes': get_num_detectors(circ),
        'avg_data_creation_rate [bit/round]': get_avg_data_creation(circ, logical_ex, d),
        'total_stab_rounds': total_stabilizer_rounds(logical_ex, d),
        'num_FT_d3_blocks': get_FT_d3_blocks(logical_ex),
        'num_nFT_d3_blocks': get_nFT_d3_blocks(logical_ex),
    }
    return ex_resources
## extract resources
d_vec = [3, 5, 7, 9]

for d in d_vec:
    print(get_resources('fact_circuit', d))
## cxreates interactive circuit
import webbrowser
import stim
def create_video(task_name, d):
    filename = f"circuits\{task_name}_d={d}_p=0.001.stim"
    with open(filename, "r") as f:
        circ = stim.Circuit(f.read())
    html_content = circ.diagram('interactive-html')._repr_html_()
    file_name = task_name+f'_{d}_interactive.html'
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