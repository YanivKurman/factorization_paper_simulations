from typing import List
from physical_level_infrastructure import Experiment, Surface,LatticeSurgery, InitialState, MeasurementBasis
from IR_level_infrastructure import Logical_Experiment
import stim

def compile_ex(logical_ex: Logical_Experiment, d: int, error_model): #convert the surface_level_experiment to the physical level
    surface_dict={}
    for surfaces in logical_ex.total_activated_qubits:
        surface_dict[(surfaces // 10, surfaces % 10)] = Surface(d)
    ex = Experiment(surface_dict, error_model)
    logical_to_physical_gates(ex, logical_ex)
    return ex


def divide_circuit_to_epochs(circuit: stim.Circuit): # each epoch is d stabilizer rounds
    epochs = []
    current_epoch = []
    for instruction in circuit:
        if instruction.name == "TICK":
            if current_epoch:
                epochs.append(current_epoch)
            current_epoch = []
        else:
            current_epoch.append(instruction)
    if current_epoch:  # Add any remaining instructions as a final epoch
        epochs.append(current_epoch)
    return epochs


def reorder_epochs_with_measurements_last(epochs: List[stim.Circuit]): #to keep fault-tolerancy all measurements should be after the stabilizer rounds
    reordered_epochs = []
    for epoch in epochs:
        # Separate measurement instructions from others
        observables=[instr for instr in epoch if instr.name == 'OBSERVABLE_INCLUDE']
        non_measurement_instr = [instr for instr in epoch if not (instr.name == "M" or instr.name =="MX" or instr.name =='OBSERVABLE_INCLUDE')]
        measurement_instr = [instr for instr in epoch if (instr.name == "M" or instr.name =="MX")]
        # Reorder the epoch by appending measurements at the end
        reordered_epoch = [non_measurement_instr,measurement_instr,observables]
        reordered_epochs.append(reordered_epoch)
    return reordered_epochs



def logical_to_physical_gates(ex: Experiment, logical_ex: Logical_Experiment): #call to the relevant physical_level gate according to the surface_level gate
    divided_circuit=divide_circuit_to_epochs(logical_ex.circ)
    ordered_epochs=reorder_epochs_with_measurements_last(divided_circuit)
    for epoch_ind, epoch in enumerate(ordered_epochs):
        for ind, inst in enumerate(epoch[0]):
            name = inst.name
            if name == 'MPP': # the compiler assumes that the pair is numbered such that the surgery measures the correct parity
                for parity_ind in range(len(inst.targets_copy()) // 3):
                    qubit0_name=inst.targets_copy()[parity_ind*3].value
                    qubit1_name=inst.targets_copy()[parity_ind*3+2].value
                    qubit0 = (qubit0_name // 10, qubit0_name % 10)
                    qubit1 = (qubit1_name // 10, qubit1_name % 10)
                    ex.initialize_surgery(qubit0, qubit1) if qubit1_name>qubit0_name else ex.initialize_surgery(qubit1, qubit0)
                continue
            qubits = []
            for targ_ind in range(len(inst.targets_copy())):
                qubits.append((inst.targets_copy()[targ_ind].value // 10, inst.targets_copy()[targ_ind].value % 10))
            for qubit in qubits:
                if qubit[0] * 10 + qubit[1] in logical_ex.rotated_initializations[epoch_ind]:
                    ex.flip_surface_orientation(qubit)
                if name == 'R':
                    ex.initialize_surface(qubit, InitialState.Z_PLUS)
                elif name == 'RX':
                    if ind<(len(epoch[0])-1):
                        if epoch[0][ind + 1].name in ['S', 'S_DAG']:
                            ex.initialize_surface(qubit, InitialState.Y_PLUS)
                            if qubit[0] * 10 + qubit[1] in logical_ex.rotated_initializations[epoch_ind]:
                                ex.surfaces[qubit[0],qubit[1]].is_rotated_S = 1
                        else:
                            ex.initialize_surface(qubit, InitialState.X_PLUS)
                    else:
                        ex.initialize_surface(qubit, InitialState.X_PLUS)
        for _ in range(ex.surfaces[list(ex.surfaces)[0]].dist):
            ex.stabilizer_round()
        for activated_surf in ex.activated_surfaces:
            if type(activated_surf) == LatticeSurgery:
                ex.terminate_surgery(activated_surf)

        for inst in epoch[1]:
            name = inst.name
            qubits = []
            for targ_ind in range(len(inst.targets_copy())):
                qubits.append((inst.targets_copy()[targ_ind].value // 10, inst.targets_copy()[targ_ind].value % 10))
            for qubit in qubits:
                if name == 'M':
                    ex.measure_surface(qubit, MeasurementBasis.Z_BASIS)
                elif name == 'MX':
                    ex.measure_surface(qubit, MeasurementBasis.X_BASIS)
        for inst in epoch[2]:
            name = inst.name
            if name == 'OBSERVABLE_INCLUDE':
                observable_list = []
                for targets in inst.targets_copy():
                    observable_list.append(len(ex.logical_measurements) + targets.value)
                ex.add_observable(observable_list)

