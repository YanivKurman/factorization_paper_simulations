# This file include the definition of an intermediate-level description of the circuit
# The description include several legal operations in the logical level and what surfaces are used to implement them
import stim
from enum import Enum
from typing import Dict, List, Literal
from collections import Counter
from itertools import chain


class InitialState(Enum):
    Z = 0
    X = 1
    S = 2
    S_DAG = 3

class MeasurementBasis(Enum):
    Z = 0
    X = 1

class CnotOrder(Enum): #describe which lattice surgery is first
    ZZXX = 0
    XXZZ = 1


class ParityBasis(Enum): # for lattice surgery
    ZZ = 0
    XX = 1
    ZX = 2
    XZ = 3

class TeleportationType(Enum):
    ZZ = 0
    XX = 1


class TeleRotationType(Enum): #teleportation and rotation (Similar to Hadamard)
    ZX = 0
    XZ = 1


class LongRangeCnot(Enum):
    ZZZZXX = 0
    ZZXZZZ = 1
    XXZZXX = 2


class Qubit: # a surface
    def __init__(self, name):
        self.name = name
        self.is_rotated = 0


class LogicalQubit:  # Represents a logical qubit in the original circuit
    def __init__(self, name):
        maximal_logical_ticks = 30  # Defines the maximum number of logical timestamps in the original circuit
        empty_list0 = [[] for _ in range(maximal_logical_ticks)]  # Initialize a list for Z frame measurements
        empty_list1 = [[] for _ in range(maximal_logical_ticks)]  # Initialize a list for X frame measurements
        self.frame_measurements = [empty_list0, empty_list1]  # Store Z and X frame measurements
        self.current_qubit = False  # Indicates the current qubit state
        self.name = name  # Name of the logical qubit
        self.is_rotated = [0] * maximal_logical_ticks  # Track if the qubit is rotated

# most probably it will make sense to have a class of operation\gates, but I am not sure how to do this.

class Logical_Experiment:  # Represents an IR description of the logical circuit and surfaces implementing it
    def __init__(self, width, height, logical_qubits: int):  # Initialize with grid dimensions and number of logical qubits
        self.qubits: Dict[tuple, Qubit] = {}  # Dictionary to hold the qubits (surfaces)
        self.circ = stim.Circuit()  # Initialize the stim circuit
        self.total_activated_qubits = set()  # Set to track activated qubits
        self.allocate_qubits(width, height)  # Allocate qubits based on the grid
        self.measurement_count = -1  # Initialize measurement count
        self.logical_qubits = [LogicalQubit(name) for name in range(logical_qubits)]  # Create logical qubits
        self.frame_propagation = []  # List to hold frame propagation data
        maximal_surface_ticks = 50  # Limit for surface-level ticks
        self.activated_qubits: List[int] = [[] for _ in range(maximal_surface_ticks)]  # Track activated qubits over ticks
        self.rotated_initializations = [[] for _ in range(maximal_surface_ticks)]  # Track rotations for specific gates

    def get_surface_tick(self):  # Count the number of "TICK" instructions in the circuit
        return sum(1 for instruction in self.circ if instruction.name == "TICK")

    def allocate_qubits(self, w: int, h: int): # Allocate qubits based on grid dimensions
        for i in range(w):
            for j in range(h):
                self.qubits[i + 10 * j] = Qubit(i + 10 * j)

    def init(self, qubit_name: int, state: InitialState):
        qubit = self.qubits[qubit_name]
        self.activated_qubits[self.get_surface_tick()].append(qubit.name)
        self.total_activated_qubits.add(qubit.name)
        if state == InitialState.Z:
            self.circ.append('R', qubit.name)
        elif state == InitialState.X:
            self.circ.append('RX', qubit.name)
        elif state == InitialState.S:
            self.circ.append('RX', qubit.name)
            self.circ.append('S', qubit.name)
        elif state == InitialState.S_DAG:
            self.circ.append('RX', qubit.name)
            self.circ.append('S_DAG', qubit.name)

    def logical_init(self, qubit_name: int, state: InitialState, logical_qubit: int):
        self.init(qubit_name, state)
        self.logical_qubits[logical_qubit].current_qubit = self.qubits[qubit_name]

    def logical_init_rot(self, qubit_name: int, state: InitialState, logical_qubit: int):
        self.init(qubit_name, state)
        qubit = self.qubits[qubit_name]
        self.logical_qubits[logical_qubit].current_qubit = qubit
        qubit.is_rotated = 1
        self.rotated_initializations[self.get_surface_tick()].append(qubit_name)

    def measure(self, qubit_name: int, basis: MeasurementBasis):
        qubit = self.qubits[qubit_name]
        self.circ.append("MX", qubit.name) if basis.value else self.circ.append("M", qubit.name)
        self.activated_qubits[self.get_surface_tick()].remove(qubit.name)
        self.measurement_count += 1
        qubit.is_rotated = 0

    def logical_measure(self, qubit_name: int, basis: MeasurementBasis, logical_qubit: int, logical_tick: int):
        self.measure(qubit_name, basis)
        self.logical_qubits[logical_qubit].frame_measurements[1 - basis.value][logical_tick].append(
            self.measurement_count)  # append to the opposite flip so that the value is the XOR of all

    def measure_parity(self, qubit1: Qubit, qubit2: Qubit, basis: ParityBasis):
        parity_target = [stim.target_x(qubit1.name)] if basis.value % 2 else [stim.target_z(qubit1.name)]
        parity_target.append(stim.target_combiner())
        parity_target.append(stim.target_x(qubit2.name)) if basis.value % 3 else parity_target.append(
            stim.target_z(qubit2.name))
        self.circ.append("MPP", parity_target)
        self.measurement_count += 1

    def CNOT(self, control_name: int, target_name: int, inter_name: int, order: CnotOrder, logical_control: int,
             logical_target: int, epoch: Literal[0, 1, 2, 3], logical_tick: int):
        inter = self.qubits[inter_name]
        qubits = [self.qubits[control_name], self.qubits[target_name]]
        logical_qubits = [self.logical_qubits[logical_control], self.logical_qubits[logical_target]]
        if epoch == 0:
            self.init(inter.name, InitialState(1 - order.value))
        elif epoch == 1:
            self.measure_parity(inter, qubits[order.value],
                                ParityBasis(order.value + 2 * qubits[order.value].is_rotated))
            logical_qubits[1 - order.value].frame_measurements[
                1 - order.value - qubits[1 - order.value].is_rotated][logical_tick].append(self.measurement_count)
        elif epoch == 2:
            self.measure_parity(inter, qubits[1 - order.value],
                                ParityBasis(1 - order.value + 2 * qubits[1 - order.value].is_rotated))
            logical_qubits[order.value].frame_measurements[order.value - qubits[order.value].is_rotated][logical_tick].append(
                self.measurement_count)
            self.frame_propagation.append((self.logical_qubits[logical_control].name,self.logical_qubits[logical_target].name, logical_tick))
            # self.append_CNOT_frame_propagation(self.logical_qubits[logical_control],
            #                                    self.logical_qubits[logical_target], logical_tick)
        elif epoch == 3:
            self.measure(inter.name, MeasurementBasis(order.value))
            logical_qubits[1 - order.value].frame_measurements[
                1 - order.value - qubits[1 - order.value].is_rotated][logical_tick].append(self.measurement_count)


    def Teleportation(self, source_name: int, target_name: int, tele_type: TeleportationType, logical_qubit: int, logical_tick: int,
                      epoch: Literal[0, 1, 2]):
        source = self.qubits[source_name]
        target = self.qubits[target_name]
        if epoch == 0:
            self.init(target.name, InitialState(1 - tele_type.value))
            target.is_rotated = source.is_rotated
            if target.is_rotated:
                surface_tick = self.get_surface_tick()
                self.rotated_initializations[surface_tick].append(target_name)
        elif epoch == 1:
            self.measure_parity(source, target, ParityBasis(tele_type.value))
            self.logical_qubits[logical_qubit].current_qubit = target
            self.logical_qubits[logical_qubit].frame_measurements[1 - tele_type.value][logical_tick].append(self.measurement_count)
        elif epoch == 2:
            self.measure(source.name, MeasurementBasis(1 - tele_type.value))
            self.logical_qubits[logical_qubit].frame_measurements[tele_type.value][logical_tick].append(self.measurement_count)

    def TeleRotation(self, source_name: int, target_name: int, tele_type: TeleRotationType, logical_qubit: int, logical_tick: int, epoch: Literal[0, 1, 2]):
        source = self.qubits[source_name]
        target = self.qubits[target_name]
        qubit=self.logical_qubits[logical_qubit]
        if epoch == 0:
            self.init(target.name, InitialState(tele_type.value))
            target.is_rotated = 1 - source.is_rotated
            if target.is_rotated:
                surface_tick = self.get_surface_tick()
                self.rotated_initializations[surface_tick].append(target_name)
        elif epoch == 1:
            self.measure_parity(source, target, ParityBasis(2 + tele_type.value))
            qubit.current_qubit = target
            qubit.frame_measurements = qubit.frame_measurements[::-1] #flipping X and Z frames
            qubit.is_rotated[logical_tick:] = [(1 - qubit.is_rotated[logical_tick - 1])]*(len(qubit.is_rotated)-logical_tick)
            qubit.frame_measurements[tele_type.value][logical_tick].append(self.measurement_count)
        elif epoch == 2:
            self.measure(source.name, MeasurementBasis(1 - tele_type.value))
            qubit.frame_measurements[1 - tele_type.value][logical_tick].append(self.measurement_count)


    def S_gate(self, target_name: int, ancilla_name: int, logical_qubit: int, epoch: Literal[0, 1, 2], logical_tick: int):
        target = self.qubits[target_name]
        ancilla = self.qubits[ancilla_name]
        if epoch == 0:
            self.init(ancilla.name, InitialState.S)
        elif epoch == 1:
            self.measure_parity(ancilla, target, ParityBasis(target.is_rotated * 2))
            self.logical_qubits[logical_qubit].frame_measurements[target.is_rotated][logical_tick].append(self.measurement_count)
            self.frame_propagation.append((logical_qubit, -1, logical_tick))
        elif epoch == 2:
            self.measure(ancilla.name, MeasurementBasis.X)
            self.logical_qubits[logical_qubit].frame_measurements[target.is_rotated][logical_tick].append(self.measurement_count)

    def S_DAG_gate(self, target_name: int, ancilla_name: int, logical_qubit: int, epoch: Literal[0, 1, 2], logical_tick: int):
        if epoch == 0:
            self.init(ancilla_name, InitialState.S_DAG)
        else:
            self.S_gate(target_name, ancilla_name, logical_qubit, epoch, logical_tick)

    def S_DAG_Rot_gate(self, target_name: int, ancilla_name: int, logical_qubit: int, epoch, logical_tick: int):
        target = self.qubits[target_name]
        ancilla = self.qubits[ancilla_name]
        ancilla.is_rotated = 1
        if epoch == 0:
            surface_tick = self.get_surface_tick()
            self.rotated_initializations[surface_tick].append(ancilla_name)
            self.init(ancilla_name, InitialState.S_DAG)
        elif epoch == 1:
            self.measure_parity(ancilla, target, ParityBasis.ZX)
            self.logical_qubits[logical_qubit].frame_measurements[1][logical_tick].append(self.measurement_count)
            self.frame_propagation.append((logical_qubit, -2, logical_tick))
        elif epoch == 2:
            self.measure(ancilla.name, MeasurementBasis.X)
            self.logical_qubits[logical_qubit].frame_measurements[1][logical_tick].append(self.measurement_count)

    def LongRangeCNOT(self, qubits: list, type: LongRangeCnot, logical_control: int, logical_target: int, epoch, logical_tick: int):
        control = self.qubits[qubits[0]]
        inter1 = self.qubits[qubits[1]]
        inter2 = self.qubits[qubits[2]]
        target = self.qubits[qubits[3]]
        if type.value == 0:  # specific case
            if epoch == 0:
                self.init(inter1.name, InitialState.X)
                self.init(inter2.name, InitialState.Z)
            elif epoch == 1:
                self.measure_parity(control, inter1, ParityBasis.ZZ)
                self.logical_qubits[logical_target].frame_measurements[1][logical_tick].append(self.measurement_count)
                self.measure_parity(inter2, target, ParityBasis.XX)
                self.logical_qubits[logical_control].frame_measurements[0][logical_tick].append(self.measurement_count)

            elif epoch == 2:
                self.measure_parity(inter2, inter1, ParityBasis.ZZ)
                self.logical_qubits[logical_target].frame_measurements[1][logical_tick].append(self.measurement_count)

                self.frame_propagation.append(
                    (self.logical_qubits[logical_control].name, self.logical_qubits[logical_target].name, logical_tick))

            elif epoch == 3:
                self.measure(inter1.name, MeasurementBasis.X)
                self.logical_qubits[logical_control].frame_measurements[0][logical_tick].append(self.measurement_count)
                self.measure(inter2.name, MeasurementBasis.X)
                self.logical_qubits[logical_control].frame_measurements[0][logical_tick].append(self.measurement_count)
        elif type.value == 1:  # specific case
            if epoch == 0:
                self.init(inter1.name, InitialState.X)
                self.init(inter2.name, InitialState.Z)
            elif epoch == 1:
                self.measure_parity(control, inter1, ParityBasis.ZZ)
                self.logical_qubits[logical_target].frame_measurements[0][logical_tick].append(self.measurement_count)
                self.measure_parity(inter2, target, ParityBasis.XZ)
                self.logical_qubits[logical_control].frame_measurements[0][logical_tick].append(self.measurement_count)
            elif epoch == 2:
                self.frame_propagation.append(
                    (self.logical_qubits[logical_control].name, self.logical_qubits[logical_target].name, logical_tick))
                self.measure_parity(inter2, inter1, ParityBasis.ZZ)
                self.logical_qubits[logical_target].frame_measurements[0][logical_tick].append(self.measurement_count)
            elif epoch == 3:
                self.measure(inter1.name, MeasurementBasis.X)
                self.measure(inter2.name, MeasurementBasis.X)
                self.logical_qubits[logical_control].frame_measurements[0][logical_tick] += [self.measurement_count - 1,
                                                                               self.measurement_count]
        elif type.value == 2:  # specific case
            if epoch == 0:
                self.init(inter1.name, InitialState.X)
                self.init(inter2.name, InitialState.Z)
            elif epoch == 1:
                self.measure_parity(inter2, target, ParityBasis.XX)
                self.logical_qubits[logical_control].frame_measurements[0][logical_tick].append(self.measurement_count)
                self.measure_parity(control, inter1, ParityBasis.ZZ)
                self.logical_qubits[logical_target].frame_measurements[1][logical_tick].append(self.measurement_count)
                self.frame_propagation.append(
                    (self.logical_qubits[logical_control].name, self.logical_qubits[logical_target].name, logical_tick))
            elif epoch == 2:
                self.measure_parity(inter2, inter1, ParityBasis.XX)
                self.logical_qubits[logical_control].frame_measurements[0][logical_tick].append(self.measurement_count)
            elif epoch == 3:
                self.measure(inter1.name, MeasurementBasis.Z)
                self.logical_qubits[logical_target].frame_measurements[1][logical_tick].append(self.measurement_count)
                self.measure(inter2.name, MeasurementBasis.Z)
                self.logical_qubits[logical_target].frame_measurements[1][logical_tick].append(self.measurement_count)

    def clean_frames(self):  # Clean frame propagation data
        for qubit in self.logical_qubits:
            for ind in [0, 1]:
                list_of_lists=qubit.frame_measurements[ind]
                flattened_list = list(chain.from_iterable(list_of_lists))
                counts = Counter(flattened_list)
                qubit.frame_measurements[ind] = [[item for item in sublist if counts[item] % 2 != 0] for sublist in list_of_lists]

    def tick(self):
        self.circ.append("TICK")
        self.activated_qubits[self.get_surface_tick()].extend(self.activated_qubits[self.get_surface_tick()-1])


    def observable_data(self, qubit: LogicalQubit, basis: MeasurementBasis):
        observable_data = []
        flattened_list = list(chain.from_iterable(qubit.frame_measurements[1 - basis.value]))
        for measurment in set(flattened_list):
            observable_data.append(stim.target_rec(measurment - self.measurement_count - 1))
        return observable_data

    def add_obserable(self, qubit_list:List[LogicalQubit], basis_list: List[MeasurementBasis], observable_indx: int):
        data=[]
        for ind,qubit in enumerate(qubit_list):
            data+=self.observable_data(qubit, basis_list[ind])
        counts = Counter(data)
        clean_data = [item for item in data if counts[item] % 2 != 0]
        self.circ.append('OBSERVABLE_INCLUDE', clean_data, observable_indx)

    def propagate_frames(self):
        sorted_frames = sorted(self.frame_propagation, key=lambda x: x[2])
        for propagation in sorted_frames:
            logical_tick=propagation[2]
            if propagation[1]+1 == 0:
                qubit = self.logical_qubits[propagation[0]]
                frames = []
                for j in range(logical_tick):
                    frames += (qubit.frame_measurements[1-qubit.is_rotated[-1]][j])
                qubit.frame_measurements[qubit.is_rotated[-1]][logical_tick] += frames
            elif propagation[1]+1 == -1:
                qubit = self.logical_qubits[propagation[0]]
                frames = []
                for j in range(logical_tick):
                    frames += (qubit.frame_measurements[qubit.is_rotated[-1]][j])
                qubit.frame_measurements[1-qubit.is_rotated[-1]][logical_tick] += frames
            else:
                for ind in [0,1]:
                    source_qubit = self.logical_qubits[propagation[ind]]
                    target_qubit = self.logical_qubits[propagation[1-ind]]
                    frames=[]
                    for j in range(logical_tick):
                        frames+=(source_qubit.frame_measurements[1-source_qubit.is_rotated[-1]-ind][j])
                    target_qubit.frame_measurements[1-target_qubit.is_rotated[-1]-ind][logical_tick]+=frames
        self.clean_frames()
    def append_CNOT_frame_propagation(self, control: Qubit, target: Qubit,logical_tick: int):
        self.frame_propagation.append(
            (control.name, target.name, 1-control.current_qubit.is_rotated, 1-target.current_qubit.is_rotated, logical_tick))
        self.frame_propagation.append(
            (target.name, control.name, target.current_qubit.is_rotated, control.current_qubit.is_rotated, logical_tick))
