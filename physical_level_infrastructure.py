import abc
import dataclasses
from enum import Enum
from functools import reduce

import stim
import numpy as np
from typing import Dict, List


#
class TileOrder:
    order_z = ['NW', 'NE', 'SW', 'SE']
    order_ᴎ = ['NW', 'SW', 'NE', 'SE']


class SurfaceOrientation(Enum):
    Z_VERTICAL_X_HORIZONTAL = 1
    X_VERTICAL_Z_HORIZONTAL = 0


class InitialState(Enum):
    Z_PLUS = 0
    X_PLUS = 1
    Z_MINUS = 2
    X_MINUS = 3
    Y_PLUS = 4
    Y_MINUS = 5


class MeasurementBasis(Enum):
    Z_BASIS = 0
    X_BASIS = 1


class SurgeryOrientation(Enum):
    VERTICAL = 0
    HORIZONTAL = 1


class SurgeryOperation(Enum):
    ZZ = 0
    XZ = 1
    ZX = 2
    XX = 3


class BaseErrorModel(abc.ABC):
    @abc.abstractmethod
    def generate_single_qubit_error(self, circ, qubits):
        pass

    def generate_two_qubit_error(self, circ, qubits):
        pass

    def generate_measurement_qubit_error(self, circ, qubits):
        pass


class NoErrorModel(BaseErrorModel):
    def generate_single_qubit_error(self, circ, qubits):
        pass

    def generate_two_qubit_error(self, circ, qubits):
        pass

    def generate_measurement_qubit_error(self, circ, qubits):
        pass


@dataclasses.dataclass
class ErrorModel(BaseErrorModel):
    single_qubit_error: float
    two_qubit_error: float
    measurement_error: float

    def generate_single_qubit_error(self, circ, qubits):
        circ.append("DEPOLARIZE1", qubits, self.single_qubit_error)

    def generate_two_qubit_error(self, circ, qubits):
        circ.append("DEPOLARIZE2", qubits, self.two_qubit_error)

    def generate_measurement_qubit_error(self, circ, qubits):
        circ.append("X_ERROR", qubits, self.measurement_error)


class BaseSurface(abc.ABC):

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.data_qubits = np.zeros((width, height), dtype=int)
        self.ancilla_qubits = np.zeros((width + 1, height + 1), dtype=int)
        self.ancilla_groups = {0: set(), 1: set(), 2: set(), 3: set(), 4: set(),
                               5: set()}  # 0= X stabilizer, 1= Z stabilizer, 2=X_left Z_right, 3=Z_left X_right, 4=Z_top X_bottom, 5=X_top Z_bottom
        self.even_tiles_order = TileOrder.order_z
        self.round = 0
        self.to_surgery_data_qubits = {'R': np.zeros((height,), dtype=int),
                                       'L': np.zeros((height,), dtype=int),
                                       'T': np.zeros((width,), dtype=int),
                                       'B': np.zeros((width,), dtype=int)}
        self.initial_state = InitialState.Z_PLUS

    @abc.abstractmethod
    def allocate_qubits(self, coord):
        pass

    def _all_active_ancillas(self):
        return reduce(lambda acc, x: acc.union(x), self.ancilla_groups.values(), set())


    def _get_target(self, ancilla_index, direction):
        # get ancilla and direction return corresponding data or none if no qubit
        if direction == 'SW':
            ret = ancilla_index[0] - 1, ancilla_index[1] - 1
        elif direction == 'NW':
            ret = ancilla_index[0] - 1, ancilla_index[1]
        elif direction == 'NE':
            ret = ancilla_index[0], ancilla_index[1]
        elif direction == 'SE':
            ret = ancilla_index[0], ancilla_index[1] - 1
        return None if ret[0] < 0 or ret[1] < 0 or ret[0] >= self.width or ret[1] >= self.height else ret # condition of the targets of gate exist or not

    def _get_ancilla_with_targets_and_op(self, epoch,
                                         stabilizer_group: int):  # gets direction of 2 qubit gate and which stabilizer_group (orientation independent), creates pair (source and target qubits)
        qubits = []
        operation = []
        my_ancillas = self.ancilla_groups[stabilizer_group]
        for ancilla in my_ancillas:
            loc = np.where(self.ancilla_qubits == ancilla)
            if (loc[0][0] + loc[1][0]) % 2 and (epoch == 3 or epoch == 4):
                direction = self.even_tiles_order[5 - epoch]
            else:
                direction = self.even_tiles_order[epoch - 2]
            target = self._get_target((loc[0][0], loc[1][0]), direction)
            if target is not None:
                qubits += ancilla, self.data_qubits[target]
            if stabilizer_group == 0 or (direction == 'NW' and (stabilizer_group == 2 or stabilizer_group == 5)) \
                    or (direction == 'NE' and (stabilizer_group == 3 or stabilizer_group == 5)) \
                    or (direction == 'SE' and (stabilizer_group == 3 or stabilizer_group == 4)) \
                    or (direction == 'SW' and (stabilizer_group == 2 or stabilizer_group == 4)):

                operation = "CX"
            else:
                operation = "CZ"
        return qubits, operation

    def _apply_two_qubit_gate_epoch(self, circ, epoch, error_model: BaseErrorModel):
        for ancilla_group in range(6):
            [qubits, operation] = self._get_ancilla_with_targets_and_op(epoch, ancilla_group)
            if len(qubits):
                circ.append(operation, qubits)
                error_model.generate_two_qubit_error(circ, qubits)


    def stabilizer_round(self, circ: stim.Circuit, epoch: int, measurements: list, error_model: BaseErrorModel):
        ancillas = self._all_active_ancillas()
        data_qubits = self.data_qubits.flatten()
        if epoch == 0:
            circ.append("R", ancillas)
            if self.round == 0:
                circ.append("R", data_qubits)
                error_model.generate_single_qubit_error(circ, data_qubits)
            error_model.generate_single_qubit_error(circ, ancillas)
        elif epoch == 1:
            circ.append("H", ancillas)
            if self.round == 0:
                if self.initial_state == InitialState.Z_MINUS or self.initial_state == InitialState.X_MINUS:
                    circ.append("X", data_qubits)
                elif self.initial_state == InitialState.X_PLUS or self.initial_state == InitialState.X_MINUS:
                    circ.append("H", data_qubits)
                error_model.generate_single_qubit_error(circ, data_qubits)
            error_model.generate_single_qubit_error(circ, ancillas)
        elif epoch < 6:
            self._apply_two_qubit_gate_epoch(circ, epoch, error_model)
        elif epoch == 6:
            circ.append("H", ancillas)
            error_model.generate_single_qubit_error(circ, ancillas)
        elif epoch == 7:
            error_model.generate_measurement_qubit_error(circ, ancillas)
            circ.append("M", ancillas)
            measurements.extend(ancillas)
            self.round += 1

    def qubit_data(self, qubit, measurements, loc):
        if len(np.where(np.array(measurements) == qubit)[0]) > -loc - 1:
            return stim.target_rec((np.where(np.array(measurements) == qubit)[0] - len(measurements))[loc])
        else:
            return None

    def add_detectors_for_all_ancillas(self, circ, measurements: list):
        for ancilla in self._all_active_ancillas():
            if self.qubit_data(ancilla, measurements, -2) is None:
                circ.append("DETECTOR", [self.qubit_data(ancilla, measurements, -1)],[ancilla//10,ancilla%10,0,0])
            else:
                circ.append("DETECTOR",
                            [self.qubit_data(ancilla, measurements, -1), self.qubit_data(ancilla, measurements, -2)],[ancilla//10,ancilla%10,0,0])

    def print_ancillas(self):
        print(np.flipud(self.ancilla_qubits.T))

    def print_data(self):
        print(np.flipud(self.data_qubits.T))

    @abc.abstractmethod
    def add_detectors(self, circ, measurements: list, error_model: BaseErrorModel):
        pass


class Surface(BaseSurface):
    def __init__(self, dist: int):
        super().__init__(dist, dist)
        self.dist = dist
        self.orientation = SurfaceOrientation.X_VERTICAL_Z_HORIZONTAL
        self.is_rotated_S=0
    def flip_orientation(self):
        if self.orientation.value:
            self.orientation = SurfaceOrientation.X_VERTICAL_Z_HORIZONTAL
        else:
            self.orientation = SurfaceOrientation.Z_VERTICAL_X_HORIZONTAL
        temp = self.ancilla_groups[0]
        self.ancilla_groups[0] = self.ancilla_groups[1]
        self.ancilla_groups[1] = temp


    def _allocate_to_surgery_data_qubits(self, name):
        dist = self.dist
        for i in range(2 * dist):
            if i < dist:
                self.to_surgery_data_qubits['R'][i % dist] = name
                self.to_surgery_data_qubits['L'][i % dist] = name - 10000
            else:
                self.to_surgery_data_qubits['T'][i % dist] = name
                self.to_surgery_data_qubits['B'][i % dist] = name - 1000
            name += 1
        return name

    def _allocate_ancillas(self, name):
        for i in range(self.dist + 1):
            for j in range(self.dist + 1):
                self.ancilla_qubits[i, j] = name
                if ((i + j) % 2 == 0 and self.orientation == SurfaceOrientation.Z_VERTICAL_X_HORIZONTAL) or (
                        ((i + j) % 2 == 1) and (self.orientation == SurfaceOrientation.X_VERTICAL_Z_HORIZONTAL)):
                    self.ancilla_groups[1].add(name)
                else:
                    self.ancilla_groups[0].add(name)
                name += 1
        to_remove = self.ancilla_qubits[0, 0::2].tolist() + self.ancilla_qubits[0::2, -1].tolist() + \
                    self.ancilla_qubits[1::2, 0].tolist() + self.ancilla_qubits[-1, 1::2].tolist()
        self.ancilla_groups[0] -= set(to_remove)
        self.ancilla_groups[1] -= set(to_remove)
        return name

    def _allocate_data_qubits(self, name):
        for i in range(self.dist):
            for j in range(self.dist):
                self.data_qubits[i, j] = name
                name += 1
        return name

    def allocate_qubits(self, coord):
        name = coord[0] * 10000 + coord[1] * 1000
        name = self._allocate_data_qubits(name)
        name = self._allocate_ancillas(name)
        name = self._allocate_to_surgery_data_qubits(name)

    def apply_injection_two_qubit_gate_epoch(self, circ, epoch, error_model):
        ancillas = self.ancilla_qubits[1][-3:]
        direction = self.even_tiles_order[(epoch - 4) % 4]
        x_ancilla = ancillas[0] if epoch > 3 else ancillas[2]
        z_ancilla=ancillas[1]
        z_ancilla_coord = np.where(self.ancilla_qubits == ancillas[1])
        target_z = self.data_qubits[self._get_target((z_ancilla_coord[0][0], z_ancilla_coord[1][0]), direction)]
        x_ancilla_coord = np.where(self.ancilla_qubits == x_ancilla)
        target_x = self.data_qubits[self._get_target((x_ancilla_coord[0][0], x_ancilla_coord[1][0]), direction)]
        if self.is_rotated_S:
            x_ancilla=self.get_rotated_ancilla_qubits([x_ancilla])[0]
            z_ancilla=self.get_rotated_ancilla_qubits([z_ancilla])[0]
            target_z=self.get_rotated_data_qubits([target_z])[0]
            target_x=self.get_rotated_data_qubits([target_x])[0]
        circ.append("CX", [x_ancilla, target_x])
        circ.append("CZ", [z_ancilla, target_z])
        error_model.generate_two_qubit_error(circ, [x_ancilla, target_x, z_ancilla, target_z])

    def injection_round(self, circ, epoch: int, measurements: list, error_model: BaseErrorModel):
        data_qubits = self.data_qubits.flatten()[[self.dist - 1, self.dist - 2, 2 * self.dist - 1, 2 * self.dist - 2]]
        ancillas = self.ancilla_qubits[1][-3:]
        if self.is_rotated_S:
            data_qubits = self.get_rotated_data_qubits(data_qubits)
            ancillas=self.get_rotated_ancilla_qubits(ancillas)
        all_qubits = np.concatenate((data_qubits, ancillas))
        if epoch == 0:
            circ.append("R", all_qubits)
            error_model.generate_single_qubit_error(circ, all_qubits)
        elif epoch == 1:
            circ.append("H", all_qubits)
            error_model.generate_single_qubit_error(circ, all_qubits)
        elif epoch < 6:
            self.apply_injection_two_qubit_gate_epoch(circ, epoch, error_model)
            if epoch == 3:
                circ.append("TICK")
                circ.append("SQRT_X", ancillas[1])
                error_model.generate_single_qubit_error(circ, ancillas[1])
        elif epoch == 6:
            circ.append("H", ancillas)
            error_model.generate_single_qubit_error(circ, ancillas)
        elif epoch == 7:
            error_model.generate_measurement_qubit_error(circ, ancillas)
            circ.append("M", ancillas)
            measurements.extend(ancillas)
            self.round += 1

    def get_rotated_ancilla_qubits(self, qubits):
        new_qubits=[]
        for qubit in qubits:
            coord = np.where(self.ancilla_qubits == qubit)
            new_qubits.append(self.ancilla_qubits[coord[1][0], self.dist - coord[0][0]])
        return new_qubits
    def get_rotated_data_qubits(self, qubits):
        new_qubits=[]
        for qubit in qubits:
            coord = np.where(self.data_qubits == qubit)
            new_qubits.append(self.data_qubits[coord[1][0], self.dist-1 - coord[0][0]])
        return new_qubits

    def get_rotated_qubits(self, qubits):
        new_qubits=[]
        for qubit in qubits:
            if qubit in self.data_qubits.flatten():
                new_qubits=new_qubits+self.get_rotated_data_qubits([qubit])
            else:
                new_qubits=new_qubits+self.get_rotated_ancilla_qubits([qubit])
        return new_qubits

    def expansion_ancilla_qubits(self, dist):
        large_dist = self.dist
        is_even = 1 - dist % 2
        qubits = list(self.ancilla_qubits[0, large_dist - 2:(large_dist - dist):-2])
        qubits += list(self.ancilla_qubits[dist, large_dist - 1 - is_even:(large_dist - dist):-2])
        for i in range(1, dist):
            if i % 2:
                qubits += list(self.ancilla_qubits[i, large_dist + 1:(large_dist - dist - is_even):-1])
            else:
                if large_dist == dist:
                    qubits += list(self.ancilla_qubits[i, large_dist - 1:0:-1])
                    qubits += list([self.ancilla_qubits[i, 0]])
                else:
                    qubits += list(self.ancilla_qubits[i, large_dist - 1:(large_dist - dist - 1 + is_even):-1])
        return qubits

    def apply_expansion_two_qubit_gate_epoch(self, circ, epoch, error_model: BaseErrorModel):
        dist = self.round + 2
        d = self.dist
        data_qubits_in_expansion_surface = list(
            self.data_qubits[0:dist, d:d - dist - 1:-1].flatten()) if dist < d else list(self.data_qubits.flatten())
        expansion_ancilla = self.expansion_ancilla_qubits(dist)
        for ancilla_group in range(6):
            [qubits, operation] = self._get_ancilla_with_targets_and_op(epoch, ancilla_group)
            if len(qubits):
                indices = [index for index, value in enumerate(qubits) if
                           value in data_qubits_in_expansion_surface and qubits[index - 1] in expansion_ancilla]
                all_indices = [val for pair in zip([x - 1 for x in indices], indices) for val in pair]
                qubits_for_operation = [qubits[i] for i in all_indices]
                if self.is_rotated_S:
                    qubits_for_operation = self.get_rotated_qubits(qubits_for_operation)
                    # if epoch in [3,4]:
                    #     operation = "CX" if operation=="CZ" else "CZ"
                circ.append(operation, qubits_for_operation)
                error_model.generate_two_qubit_error(circ, qubits_for_operation)

    def expansion_round(self, circ, epoch: int, measurements: list, error_model: BaseErrorModel):
        dist = self.round + 2
        d = self.dist
        data_qubits_bottom = self.data_qubits[0:dist, d - dist]
        data_qubits_right = self.data_qubits[dist - 1, d:d - dist:-1]
        ancillas = self.expansion_ancilla_qubits(self.round + 2)
        if self.is_rotated_S:
            data_qubits_bottom=self.get_rotated_data_qubits(data_qubits_bottom)
            data_qubits_right = self.get_rotated_data_qubits(data_qubits_right)
            ancillas= self.get_rotated_ancilla_qubits(ancillas)
        new_data_qubits = np.concatenate((data_qubits_bottom, data_qubits_right))
        if epoch == 0:
            circ.append("R", np.concatenate((new_data_qubits, ancillas)))
            error_model.generate_single_qubit_error(circ, np.concatenate((new_data_qubits, ancillas)))
        elif epoch == 1:
            qubits=data_qubits_bottom
            circ.append("H", np.concatenate((qubits, ancillas)))
            error_model.generate_single_qubit_error(circ, np.concatenate((data_qubits_bottom, ancillas)))
        elif epoch < 6:
            self.apply_expansion_two_qubit_gate_epoch(circ, epoch, error_model)
        elif epoch == 6:
            circ.append("H", ancillas)
            error_model.generate_single_qubit_error(circ, ancillas)
        elif epoch == 7:
            error_model.generate_measurement_qubit_error(circ, ancillas)
            circ.append("M", ancillas)
            measurements.extend(ancillas)
            self.round += 1

    def add_measurement_detectors(self, circ: stim.Circuit, basis: MeasurementBasis, measurements: list):
        stabilizer_group = 0 if basis == MeasurementBasis.X_BASIS else 1
        ancilla_target_list = []
        for epoch in [2, 3, 4, 5]:
            ancilla_target_list += self._get_ancilla_with_targets_and_op(epoch, stabilizer_group)[0]
        ancila_target_list = list(set(ancilla_target_list))
        ancillas = sorted(i for i in ancila_target_list if i > self.data_qubits[-1][-1])
        for ancilla in ancillas:
            locs = np.where(np.array(ancilla_target_list) == ancilla)[0]
            target = np.array(ancilla_target_list)[locs + 1]
            if len(target) == 2:
                circ.append("DETECTOR",
                            [self.qubit_data(ancilla, measurements, -1), self.qubit_data(target[0], measurements, -1),
                             self.qubit_data(target[1], measurements, -1)],[ancilla//10,ancilla%10,0,0])
            else:
                circ.append("DETECTOR",
                            [self.qubit_data(ancilla, measurements, -1), self.qubit_data(target[0], measurements, -1),
                             self.qubit_data(target[1], measurements, -1), \
                             self.qubit_data(target[2], measurements, -1),
                             self.qubit_data(target[3], measurements, -1)],[ancilla//10,ancilla%10,0,0])

    def observable_data(self, measurements: list, basis: MeasurementBasis):
        dist = self.dist
        observable_qubits = self.data_qubits[0:dist, 0] if self.orientation.value == basis.value else self.data_qubits[
                                                                                                      0, 0:dist]
        observable_data = []
        for qubits in observable_qubits.flatten():
            observable_data.append(self.qubit_data(qubits, measurements, -1))
        return observable_data

    def add_observable(self, circ: stim.Circuit, measurements: list, basis: MeasurementBasis, observable_index: int):
        circ.append('OBSERVABLE_INCLUDE', self.observable_data(measurements, basis), observable_index)

    # def apply_feedback(self, circ: stim.Circuit, observable_data, feedback: MeasurementBasis,
    #                    error_model: BaseErrorModel): #not used
    #     dist = self.dist
    #     target_qubits = self.data_qubits[0:dist, 0] if self.orientation.value == feedback.value else self.data_qubits[
    #                                                                                                  0, 0:dist]
    #     for qubit in target_qubits:
    #         for data in observable_data:
    #             circ.append("CX", [data, qubit]) if feedback == MeasurementBasis.X_BASIS else circ.append("CZ",
    #                                                                                                       [data, qubit])
    #     error_model.generate_single_qubit_error(circ, target_qubits)

    def surface_measurement(self, circ: stim.Circuit, basis: MeasurementBasis, error_model: BaseErrorModel,
                            measurements: list):
        data_qubits = self.data_qubits.flatten()
        if basis == MeasurementBasis.X_BASIS:
            circ.append('H', data_qubits)
            error_model.generate_single_qubit_error(circ, data_qubits)
            circ.append("Tick")
        error_model.generate_measurement_qubit_error(circ, data_qubits)
        circ.append('MZ', data_qubits)
        measurements.extend(data_qubits)
        self.round = 0
        self.add_measurement_detectors(circ, basis, measurements)

    def add_surface_initialization_detectors(self, circ, measurements: list):
        state=self.initial_state
        if state== InitialState.Z_PLUS or state == InitialState.Z_MINUS:
            ancillas_for_detectors = self.ancilla_groups[1]
        elif state == InitialState.X_PLUS or state == InitialState.X_MINUS:
            ancillas_for_detectors = self.ancilla_groups[0]
        elif state == InitialState.Y_PLUS:
            ancillas_for_detectors = self.ancilla_qubits[1][-3::2]
            if self.is_rotated_S:
                ancillas_for_detectors=self.get_rotated_ancilla_qubits(ancillas_for_detectors)
        for ancilla in ancillas_for_detectors:
            circ.append("DETECTOR", [self.qubit_data(ancilla, measurements, -1)],[ancilla//10,ancilla%10,1, 1] if state == InitialState.Y_PLUS else [ancilla//10,ancilla%10,1,0])

    def feedback_target(self, ancilla): #for keeping stabilizers +1
        (x, y) = np.where(self.ancilla_qubits == ancilla)
        x = x[0]
        y = y[0]
        d = self.dist
        d2 = (d + 1) / 2
        if y == 0:
            return self.data_qubits.T[0][0:x] if x < d2 else self.data_qubits.T[0][x:d]
        if x == 0:
            return self.data_qubits[0][0:y] if y < d2 else self.data_qubits[0][y:d]
        if (x + y) % 2 == 0:
            return self.data_qubits.T[y - 1][0:x] if x < d2 else self.data_qubits.T[y - 1][x:d]
        else:
            return self.data_qubits[x - 1][0:y] if y < d2 else self.data_qubits[x - 1][y:d]

    def get_ancillas_for_expansion_feedback(self):
        dist = self.round + 1
        d = self.dist
        ancillas_for_feedback_CX = self.ancilla_qubits[(1 - dist % 2):dist:2, d - dist + 1].tolist()
        ancillas_for_feedback_CZ = self.ancilla_qubits[dist - 1, d - dist + 2:d - dist % 2 + 1:2].tolist()
        return ancillas_for_feedback_CZ, ancillas_for_feedback_CX

    def add_surface_expansion_detectors(self, circ, measurements: list):
        ancillas_for_feedback_CX, ancillas_for_feedback_CZ = self.get_ancillas_for_expansion_feedback()
        for ancilla in [item for item in self.expansion_ancilla_qubits(self.round + 1) if
                        item not in ancillas_for_feedback_CX + ancillas_for_feedback_CZ]:
            if self.is_rotated_S:
                ancilla = self.get_rotated_ancilla_qubits([ancilla])[0]
            if self.qubit_data(ancilla, measurements, -2) is None:
                circ.append("DETECTOR", [self.qubit_data(ancilla, measurements,-1)],[ancilla//10,ancilla%10,0,1])
            else:
                circ.append("DETECTOR",
                            [self.qubit_data(ancilla, measurements, -1), self.qubit_data(ancilla, measurements, -2)],[ancilla//10,ancilla%10,0,0])

    def add_expansion_feedback(self, circ, measurements: list, error_model: BaseErrorModel):
        ancillas_for_feedback_CZ, ancillas_for_feedback_CX = self.get_ancillas_for_expansion_feedback()
        all_target = []
        for ancilla in ancillas_for_feedback_CX + ancillas_for_feedback_CZ:
            ancilla_coord = np.where(self.ancilla_qubits == ancilla)
            target_qubit = self.data_qubits[ancilla_coord[0], ancilla_coord[1] - 1]
            command = "CX" if ancilla in ancillas_for_feedback_CX else "CZ"
            if self.is_rotated_S:
                ancilla = self.get_rotated_ancilla_qubits([ancilla])[0]
                target_qubit = self.get_rotated_data_qubits([target_qubit])[0]
            circ.append(command, [self.qubit_data(ancilla, measurements, -1), target_qubit])
            all_target += target_qubit
            measurements[np.where(np.array(measurements) == ancilla)[0][0]] = -1
        error_model.generate_single_qubit_error(circ, all_target)

    def add_initialization_feedback(self, circ, measurements: list, error_model: BaseErrorModel): #for keeping stabilizers +1
        if self.initial_state == InitialState.Z_PLUS or self.initial_state == InitialState.Z_MINUS:
            ancillas_for_feedback = self.ancilla_groups[0]
            command = "CZ"
        elif self.initial_state == InitialState.X_PLUS or self.initial_state == InitialState.X_MINUS:
            ancillas_for_feedback = self.ancilla_groups[1]
            command = "CX"
        elif self.initial_state == InitialState.Y_PLUS:
            ancillas_for_feedback = [self.ancilla_qubits[1][-2]]
            command = "CX"
        all_target = []
        for ancilla in ancillas_for_feedback:
            target_qubits = self.feedback_target(ancilla)
            if self.is_rotated_S:
                target_qubits=self.get_rotated_data_qubits(target_qubits)
                ancilla=self.get_rotated_ancilla_qubits([ancilla])[0]
            for qubit in target_qubits:
                circ.append(command, [self.qubit_data(ancilla, measurements, -1), qubit])
                all_target.append(qubit)
            measurements[np.where(np.array(measurements) == ancilla)[0][0]] = -1
        error_model.generate_single_qubit_error(circ, set(all_target))

    def add_detectors(self, circ, measurements: list, error_model: BaseErrorModel):
        if self.round == 1:
            self.add_surface_initialization_detectors(circ, measurements)
            self.add_initialization_feedback(circ, measurements, error_model)
        elif self.round > 1:
            if self.initial_state == InitialState.Y_PLUS and self.round < self.dist:
                self.add_surface_expansion_detectors(circ, measurements)
                self.add_expansion_feedback(circ, measurements, error_model)
            else:
                self.add_detectors_for_all_ancillas(circ, measurements)

    def print_surface_name(self):
        print(self.data_qubits)

    def verify_y(self, circ: stim.Circuit):
        dist = self.dist
        observable_qubits_z = self.data_qubits[1:dist, 0] if 1 - self.orientation.value else self.data_qubits[0, 1:dist]
        observable_qubits_x = self.data_qubits[1:dist, 0] if self.orientation.value else self.data_qubits[0, 1:dist]
        parity_target = [stim.target_y(self.data_qubits[0, 0])]
        for qubit in observable_qubits_z:
            parity_target.append(stim.target_combiner())
            parity_target.append(stim.target_z(qubit))
        for qubit in observable_qubits_x:
            parity_target.append(stim.target_combiner())
            parity_target.append(stim.target_x(qubit))
        circ.append("MPP", parity_target)


class LatticeSurgery(BaseSurface):

    def __init__(self, surface1: Surface, surface2: Surface, surgery_orientation: SurgeryOrientation):
        super().__init__(
            surface1.dist + surface2.dist + 1 if surgery_orientation == SurgeryOrientation.HORIZONTAL else surface1.dist,
            surface1.dist + surface2.dist + 1 if surgery_orientation == SurgeryOrientation.VERTICAL else surface1.dist
        )
        self.surface1 = surface1
        self.surface2 = surface2
        self.orientation = surgery_orientation
        if surface1.dist != surface2.dist:
            raise RuntimeError("Surfaces should be with the same dist")
        self.surgery_data_qubits = surface1.to_surgery_data_qubits[
            'T'] if surgery_orientation == SurgeryOrientation.VERTICAL else surface1.to_surgery_data_qubits['R']

    def _surgery_operation(self):
        return SurgeryOperation((self.surface1.orientation.value + self.surface2.orientation.value * 2) * (
                1 - self.orientation.value) + self.orientation.value * ((1 - self.surface1.orientation.value) + (
                1 - self.surface2.orientation.value) * 2))

    def _allocate_data_qubits(self):
        dist = self.surface1.dist
        self.data_qubits[0:dist, 0:dist] = self.surface1.data_qubits
        if self.orientation == SurgeryOrientation.HORIZONTAL:
            self.data_qubits[(dist + 1):(2 * dist + 1), 0:dist] = self.surface2.data_qubits
            self.data_qubits[dist, 0:dist] = self.surgery_data_qubits
        else:
            self.data_qubits[0:dist, (dist + 1):(2 * dist + 1)] = self.surface2.data_qubits
            self.data_qubits[0:dist, dist] = self.surgery_data_qubits
        self.round = 1  # assuming that we do not want to initialize an alongated surface.

    def _allocate_ancillas(self):
        self.ancilla_groups[0] = self.surface1.ancilla_groups[0].union(self.surface2.ancilla_groups[0])
        self.ancilla_groups[1] = self.surface1.ancilla_groups[1].union(self.surface2.ancilla_groups[1])
        dist = self.surface1.dist
        self.ancilla_qubits[0:dist + 1, 0:dist + 1] = self.surface1.ancilla_qubits
        op_val = self._surgery_operation().value
        if self.orientation == SurgeryOrientation.HORIZONTAL:
            self.ancilla_qubits[dist + 1:, 0:dist + 1] = self.surface2.ancilla_qubits
            self.ancilla_groups[self.surface1.orientation.value].update(self.surface1.ancilla_qubits[-1, 1::2])
            if not (op_val % 3):
                self.ancilla_groups[self.surface2.orientation.value].update(self.surface2.ancilla_qubits[0, 0::2])
            else:
                self.ancilla_groups[1 + op_val].update(self.surface2.ancilla_qubits[0, 0::2])
                self.ancilla_groups[4 - op_val].update(self.surface2.ancilla_qubits[0, 1:-1:2])
                self.ancilla_groups[1 - self.surface2.orientation.value] -= set(self.surface2.ancilla_qubits[0, 1::2])
        elif self.orientation == SurgeryOrientation.VERTICAL:
            self.ancilla_qubits[0:dist + 1, dist + 1:] = self.surface2.ancilla_qubits
            self.ancilla_groups[1 - self.surface1.orientation.value].update(self.surface1.ancilla_qubits[0::2, -1])
            if not (op_val % 3):
                self.ancilla_groups[1 - self.surface2.orientation.value].update(self.surface2.ancilla_qubits[1::2, 0])
            else:
                self.ancilla_groups[3 + op_val].update(self.surface2.ancilla_qubits[1::2, 0])
                self.ancilla_groups[6 - op_val].update(self.surface2.ancilla_qubits[2::2, 0])
                self.ancilla_groups[self.surface2.orientation.value] -= set(self.surface2.ancilla_qubits[2::2, 0])

    def _allocate_to_surgery_data_qubits(self):
        if self.orientation == SurgeryOrientation.HORIZONTAL:
            self.to_surgery_data_qubits['L'] = self.surface1.to_surgery_data_qubits['L']
            self.to_surgery_data_qubits['R'] = self.surface2.to_surgery_data_qubits['R']
            self.to_surgery_data_qubits['T'][0:self.surface1.width] = self.surface1.to_surgery_data_qubits['T']
            self.to_surgery_data_qubits['T'][self.surface1.width] = max(self.surface1.to_surgery_data_qubits['T']) + 1
            self.to_surgery_data_qubits['T'][self.surface1.width + 1:self.surface1.width + self.surface2.width + 1] = \
                self.surface2.to_surgery_data_qubits['T']
            self.to_surgery_data_qubits['B'][0:self.surface1.width] = self.surface1.to_surgery_data_qubits['B']
            self.to_surgery_data_qubits['B'][self.surface1.width] = max(
                self.surface1.to_surgery_data_qubits['T']) + 1 - 1000
            self.to_surgery_data_qubits['B'][self.surface1.width + 1:self.surface1.width + self.surface2.width + 1] = \
                self.surface2.to_surgery_data_qubits['B']
        else:
            self.to_surgery_data_qubits['B'] = self.surface1.to_surgery_data_qubits['B']
            self.to_surgery_data_qubits['T'] = self.surface2.to_surgery_data_qubits['T']
            self.to_surgery_data_qubits['R'][0:self.surface1.height] = self.surface1.to_surgery_data_qubits['R']
            self.to_surgery_data_qubits['R'][self.surface1.height] = max(self.surface1.to_surgery_data_qubits['T']) + 1
            self.to_surgery_data_qubits['R'][self.surface1.height + 1:self.surface1.dist + self.surface2.height + 1] = \
                self.surface2.to_surgery_data_qubits['R']
            self.to_surgery_data_qubits['L'][0:self.surface1.height] = self.surface1.to_surgery_data_qubits['L']
            self.to_surgery_data_qubits['L'][self.surface1.height] = max(
                self.surface1.to_surgery_data_qubits['T']) + 1 - 10000
            self.to_surgery_data_qubits['L'][self.surface1.height + 1:self.surface1.height + self.surface2.height + 1] = \
                self.surface2.to_surgery_data_qubits['L']

    def allocate_qubits(self, coord):
        self._allocate_data_qubits()
        self._allocate_ancillas()
        self._allocate_to_surgery_data_qubits()

    def initialize_surgery_data(self, circ: stim.Circuit, error_model: BaseErrorModel):
        circ.append("R", self.surgery_data_qubits)
        if not (self._surgery_operation().value % 2):
            circ.append("H", self.surgery_data_qubits)
        error_model.generate_single_qubit_error(circ, self.surgery_data_qubits)

    def observable_qubits(self):
        dist = self.surface1.dist
        return np.concatenate((self.ancilla_qubits[dist, 1::2], self.ancilla_qubits[dist + 1,
                                                                0::2])).flatten() if self.orientation == SurgeryOrientation.HORIZONTAL else \
            np.concatenate((self.ancilla_qubits[0::2, dist], self.ancilla_qubits[1::2, dist + 1])).flatten()

    def add_surgery_initialization_detectors(self, circ: stim.Circuit, measurements: list):
        ancillas_for_detection = self._all_active_ancillas() - set(self.observable_qubits())
        for ancilla in ancillas_for_detection:
            circ.append("DETECTOR",
                        [self.qubit_data(ancilla, measurements, -1), self.qubit_data(ancilla, measurements, -2)],[ancilla//10,ancilla%10,0,0])

    def add_surgery_initialization_feedback(self, circ, measurements, error_model: BaseErrorModel):
        dist = self.surface1.dist
        operation = self._surgery_operation()
        for j in range(0, dist, 2):
            qubits = self.data_qubits[dist:2 * dist + 1,
                     j] if self.orientation == SurgeryOrientation.HORIZONTAL else self.data_qubits[j, dist:2 * dist + 1]
            ancilla = self.ancilla_qubits[dist + 1, j] if self.orientation == SurgeryOrientation.HORIZONTAL else \
                self.ancilla_qubits[j + 1, dist + 1]
            ancilla2 = self.ancilla_qubits[dist, j + 1] if self.orientation == SurgeryOrientation.HORIZONTAL else \
                self.ancilla_qubits[j, dist]
            for i, qubit in enumerate(qubits):
                if i == 0:
                    command = "CZ" if operation.value % 2 else "CX"
                    circ.append(command, [self.qubit_data(ancilla2, measurements, -1), qubit])
                else:
                    command = "CZ" if operation.value > 1 else "CX"
                    circ.append(command, [self.qubit_data(ancilla, measurements, -1), qubit])
                    circ.append(command, [self.qubit_data(ancilla2, measurements, -1), qubit])
            measurements[np.where(np.array(measurements) == ancilla)[0][0]] = -1
            measurements[np.where(np.array(measurements) == ancilla2)[0][0]] = -1
            error_model.generate_single_qubit_error(circ, qubits)

    def observable_data(self, measurements: list):
        observable_data = []
        for qubits in self.observable_qubits():
            observable_data.append(self.qubit_data(qubits, measurements, -1))
        return observable_data

    def add_observable(self, circ: stim.Circuit, measurements, observable):
        circ.append('OBSERVABLE_INCLUDE', self.observable_data(measurements), observable)

    def add_detectors(self, circ, measurements: list, error_model: BaseErrorModel):
        if self.round == 2:
            self.add_surgery_initialization_detectors(circ, measurements)
            # self.add_surgery_initialization_feedback(circ, measurements, error_model)
        else:
            self.add_detectors_for_all_ancillas(circ, measurements)

    def add_surgery_measurement_feedback(self, circ, measurements: list, error_model: BaseErrorModel):
        dist = self.surface1.dist
        control_qubits = self.surgery_data_qubits
        for j, qubit in enumerate(control_qubits):
            if self.orientation == SurgeryOrientation.HORIZONTAL:
                target_qubits = np.concatenate([self.data_qubits[dist - 1, 0:j], self.data_qubits[dist + 1, 0:j],
                                                [self.data_qubits[dist + (-1) ** j, j]]]) if j < (dist + 1) / 2 \
                    else np.concatenate([self.data_qubits[dist - 1, j + 1:], self.data_qubits[dist + 1, j + 1:],
                                         [self.data_qubits[dist + (-1) ** (j + 1), j]]])
            else:
                target_qubits = np.concatenate([self.data_qubits[0:j, dist - 1], self.data_qubits[0:j, dist + 1],
                                                [self.data_qubits[j, dist + (-1) ** (j + 1)]]]) if j < (dist - 1) / 2 \
                    else np.concatenate([self.data_qubits[j + 1:, dist - 1], self.data_qubits[j + 1:, dist + 1],
                                         [self.data_qubits[j, dist + (-1) ** j]]])
            for target in target_qubits:
                if target < qubit:
                    command = "CX" if self._surgery_operation().value % 2 else "CZ"
                else:
                    command = "CX" if self._surgery_operation().value > 1 else "CZ"
                circ.append(command, [self.qubit_data(qubit, measurements, -1), target])
            error_model.generate_single_qubit_error(circ, target_qubits)

    def measure_surgery_data(self, circ, measurements, error_model: BaseErrorModel):
        error_model.generate_measurement_qubit_error(circ, self.surgery_data_qubits)
        if not (self._surgery_operation().value % 2):
            circ.append("H", self.surgery_data_qubits)
        circ.append("M", self.surgery_data_qubits)
        measurements.extend(self.surgery_data_qubits)
        self.add_surgery_measurement_feedback(circ, measurements, error_model)


class Experiment:

    def __init__(self, surfaces: Dict[tuple, Surface], error_model: BaseErrorModel):
        self.surfaces = surfaces
        self.circ = stim.Circuit()
        self.surgeries: Dict[tuple, LatticeSurgery] = {}
        for coordinate, surface in surfaces.items():
            surface.allocate_qubits(coordinate)

        for coordinate, surface in surfaces.items(): #add only if we want to view
            self._allocate_qubit_coordinates(self.circ, coordinate)
        for coordinate, surface in surfaces.items():
            self._allocate_surgery(surface, coordinate, SurgeryOrientation.HORIZONTAL)
            self._allocate_surgery(surface, coordinate, SurgeryOrientation.VERTICAL)
        self.activated_surfaces: List[BaseSurface] = []
        self.measurements = []
        self.error_model = error_model
        self.observable_index = 0
        self.logical_measurements = []
        self.physical_qubits=set()

    def _allocate_qubit_coordinates(self, circ, coordinate):
        d = self.surfaces[coordinate].dist
        for i in range(d):
            for j in range(d):
                circ.append('QUBIT_COORDS', self.surfaces[coordinate].data_qubits[i, j],
                            (i + 0.5 + (d + 1) * coordinate[0], 0.5 + j + (d + 1) * coordinate[1]))
            circ.append('QUBIT_COORDS', self.surfaces[coordinate].to_surgery_data_qubits['R'][i],
                        (d + 0.5 + (d + 1) * coordinate[0], 0.5 + i + (d + 1) * coordinate[1]))
            circ.append('QUBIT_COORDS', self.surfaces[coordinate].to_surgery_data_qubits['T'][i],
                        (i + 0.5 + (d + 1) * coordinate[0], 0.5 + d + (d + 1) * coordinate[1]))
        for i in range(d + 1):
            for j in range(d + 1):
                circ.append('QUBIT_COORDS', self.surfaces[coordinate].ancilla_qubits[i, j],
                            (i + (d + 1) * coordinate[0], j + (d + 1) * coordinate[1]))

    def _allocate_surgery(self, surface, coordinate, orientation: SurgeryOrientation):
        other_coord = (coordinate[0], coordinate[1] + 1) if orientation == SurgeryOrientation.VERTICAL else (
            coordinate[0] + 1, coordinate[1])
        if other_coord not in self.surfaces:
            return
        surgery = LatticeSurgery(surface, self.surfaces[other_coord], orientation)
        self.surgeries[coordinate, other_coord] = surgery

    def activate_surface(self, surface: BaseSurface):
        if isinstance(surface, Surface):
            self.activated_surfaces = [x for x in self.activated_surfaces if
                                       (isinstance(x, Surface) or (x.surface1 != surface and x.surface2 != surface))]
            self.activated_surfaces.append(surface)
        elif isinstance(surface, LatticeSurgery):
            self.activated_surfaces = [x for x in self.activated_surfaces if
                                       (isinstance(x, LatticeSurgery) or (
                                               x != surface.surface1 and x != surface.surface2))]
            self.activated_surfaces.append(surface)

    def __getitem__(self, coor):
        return self.surfaces[coor]

    def flip_surface_orientation(self, coor: tuple):
        self.surfaces[coor].flip_orientation()


    def measure_surface(self, coor: tuple, basis: MeasurementBasis):
        self.surfaces[coor].surface_measurement(self.circ, basis, self.error_model, self.measurements)
        self.add_logical_measurement(self.surfaces[coor], basis)
        self.activated_surfaces= [x for x in self.activated_surfaces if x != self.surfaces[coor]]
        self.remove_surface_measurements(self.surfaces[coor])

    def remove_surface_measurements(self, surface: Surface):
        self.measurements = [-1 if item in surface.data_qubits.flatten() else item for item in
                             self.measurements]
        self.measurements = [-1 if item in surface.ancilla_qubits.flatten() else item for item in
                             self.measurements]

    def initialize_surface(self, coor: tuple, state: InitialState):
        self.activate_surface(self.surfaces[coor])
        self.physical_qubits.update(self.surfaces[coor].data_qubits.flatten())
        self.physical_qubits.update(set.union(*self.surfaces[coor].ancilla_groups.values()))
        self.surfaces[coor].initial_state = state

    def stabilizer_round(self):
        for epoch in range(8):
            for surface in self.activated_surfaces:
                if surface.initial_state == InitialState.Y_PLUS:
                    if surface.round == 0:
                        surface.injection_round(self.circ, epoch, self.measurements, self.error_model)
                    elif surface.round < (
                            surface.dist - 1):
                        surface.expansion_round(self.circ, epoch, self.measurements, self.error_model)
                    else:
                        surface.stabilizer_round(self.circ, epoch, self.measurements, self.error_model)
                else:
                    surface.stabilizer_round(self.circ, epoch, self.measurements, self.error_model)
            self.circ.append("TICK")
        for surface in self.activated_surfaces:
            if isinstance(surface, LatticeSurgery) and surface.round == 2:
                self.add_logical_surgery_measurement(surface)
            surface.add_detectors(self.circ, self.measurements, self.error_model)
            if isinstance(surface, Surface):
                if surface.round == surface.dist - 1 and surface.is_rotated_S:
                    surface.is_rotated_S = 0
                    surface.flip_orientation()

    def initialize_surgery(self, coord0: tuple, coord1: tuple):
        surgery = self.surgeries[(coord0, coord1)]
        surgery.allocate_qubits(coord0)
        self.activate_surface(surgery)
        self.physical_qubits.update(surgery.surgery_data_qubits.ravel())
        self.physical_qubits.update(set.union(*surgery.ancilla_groups.values()))
        surgery.initialize_surgery_data(self.circ, self.error_model)

    def terminate_surgery(self, surgery: LatticeSurgery):
        surgery.measure_surgery_data(self.circ, self.measurements, self.error_model)
        self.activate_surface(surgery.surface1)
        self.activate_surface(surgery.surface2)

    def verify_y_state(self, coor: tuple):
        self.surfaces[coor].verify_y(self.circ)
        self.circ.append('OBSERVABLE_INCLUDE', stim.target_rec(-1), self.observable_index)
        self.observable_index += 1

    def add_logical_measurement(self, surface: Surface, basis: MeasurementBasis):
        offset = len(self.measurements)
        self.logical_measurements.append([surface.observable_data(self.measurements, basis), offset])

    def add_logical_surgery_measurement(self, surgery: LatticeSurgery):
        self.logical_measurements.append([surgery.observable_data(self.measurements), len(self.measurements)])

    def update_logical_measurements(self, ind: int):
        num_meas = len(self.measurements)
        offset = self.logical_measurements[ind][1]
        for meas_ind, meas in enumerate(self.logical_measurements[ind][0]):
            new_value = -(-meas.value + num_meas - offset)
            self.logical_measurements[ind][0][meas_ind] = stim.target_rec(new_value)
        self.logical_measurements[ind][1]=len(self.measurements)
        return self.logical_measurements[ind][0]

    def add_observable(self, logical_meas_index: List[int]):
        observable_data = []
        for ind in logical_meas_index:
            observable_data += self.update_logical_measurements(ind)
        self.circ.append('OBSERVABLE_INCLUDE', observable_data, self.observable_index)
        self.observable_index += 1
