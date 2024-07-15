from IR_level_infrastructure import Logical_Experiment, InitialState,CnotOrder,TeleportationType, LongRangeCnot, MeasurementBasis

 ## up to the end https://algassert.com/quirk#circuit={%22cols%22:[[%22H%22,1,%22H%22,%22H%22],[%22Z^%C2%BD%22,1,%22Z^-%C2%BD%22,%22Z^-%C2%BD%22],[%22%E2%80%A2%22,1,1,%22X%22],[1,1,1,%22Z^%C2%BD%22],[1,1,%22%E2%80%A2%22,%22X%22],[1,1,1,%22Z^-%C2%BD%22],[%22%E2%80%A2%22,1,%22X%22],[%22%E2%80%A2%22,1,1,%22X%22],[1,1,%22Z^-%C2%BD%22,%22Z^%C2%BD%22,%22H%22],[%22%E2%80%A2%22,1,%22X%22],[1,%22H%22,1,%22X^%C2%BD%22,%22Z^-%C2%BD%22],[%22X%22,%22%E2%80%A2%22],[1,1,1,%22Z^%C2%BD%22],[%22%E2%80%A2%22,1,1,1,%22X%22],[1,1,%22Z^%C2%BD%22,1,%22Z^%C2%BD%22],[%22Z^%C2%BD%22],[1,1,1,%22%E2%80%A2%22,%22X%22],[1,1,1,1,%22Z^-%C2%BD%22],[%22%E2%80%A2%22,1,1,%22X%22],[%22%E2%80%A2%22,1,1,1,%22X%22],[1,1,1,1,%22Z^%C2%BD%22],[1,1,1,%22Z^-%C2%BD%22,%22X^%C2%BD%22],[%22%E2%80%A2%22,1,1,%22X%22],[1,1,1,%22H%22,%22Z^%C2%BD%22],[1,1,1,%22Measure%22],[%22%E2%80%A2%22,1,1,1,%22X%22],[1,1,%22%E2%80%A2%22,1,%22X%22],[1,%22%E2%80%A2%22,%22X%22],[1,1,1,1,1,%22|0%E2%9F%A9%E2%9F%A80|%22],[1,1,1,1,%22%E2%80%A2%22,1,1,1,1,%22X%22],[1,%22H%22],[1,%22%E2%80%A2%22,1,1,1,1,1,1,1,%22X%22],[%22%E2%80%A2%22,1,1,1,1,1,1,1,1,%22X%22],[%22H%22,1,%22H%22],[%22%E2%80%A2%22,1,1,1,1,1,%22X%22],[1,1,1,1,%22H%22],[1,1,1,1,%22%E2%80%A2%22,1,%22X%22],[1,%22%E2%80%A2%22,1,1,1,1,1,%22X%22],[1,1,%22%E2%80%A2%22,1,1,1,1,%22X%22]]}
def build_factorization_circuit():

    logical_ex = Logical_Experiment(width=5, height=5, logical_qubits=5)
    # T=0
    logical_ex.logical_init(32, InitialState.S, logical_qubit=0)
    logical_ex.CNOT(32, 21, 31, CnotOrder.ZZXX, 0, 3, epoch=0, logical_tick=0)
    logical_ex.tick()

    # T=1
    logical_ex.CNOT(32, 21, 31, CnotOrder.ZZXX, 0, 3, epoch=1, logical_tick=0)
    logical_ex.logical_init(21, InitialState.S_DAG, logical_qubit=3)
    logical_ex.tick()

    # T=2
    logical_ex.CNOT(32, 21, 31, CnotOrder.ZZXX, 0, 3, epoch=2, logical_tick=0)
    logical_ex.Teleportation(32, 33, TeleportationType.ZZ, 0, epoch=0, logical_tick=1)
    logical_ex.Teleportation(21, 11, TeleportationType.XX, 3, epoch=0, logical_tick=1)
    logical_ex.tick()

    # T=3
    logical_ex.S_gate(11, 10, 3, epoch=0, logical_tick=2)
    logical_ex.Teleportation(21, 11, TeleportationType.XX, 3, epoch=1, logical_tick=1)
    logical_ex.Teleportation(32, 33, TeleportationType.ZZ, 0, epoch=1, logical_tick=1)
    logical_ex.CNOT(32, 21, 31, CnotOrder.ZZXX, 0, 3, epoch=3, logical_tick=0)
    logical_ex.tick()

    # T=4
    logical_ex.S_gate(11, 10, 3, epoch=1, logical_tick=2)
    logical_ex.Teleportation(32, 33, TeleportationType.ZZ, 0, epoch=2, logical_tick=1)
    logical_ex.Teleportation(21, 11, TeleportationType.XX, 3, epoch=2, logical_tick=1)
    logical_ex.tick()

    # T=5
    logical_ex.logical_init(22, InitialState.S_DAG, 2)
    logical_ex.CNOT(22, 11, 21, CnotOrder.ZZXX, 2, 3, epoch=0, logical_tick=3)
    logical_ex.CNOT(33, 22, 32, CnotOrder.ZZXX, 0, 2, epoch=0, logical_tick=4)
    logical_ex.S_gate(11, 10, 3, epoch=2, logical_tick=2)
    logical_ex.tick()

    # T=6
    logical_ex.CNOT(22, 11, 21, CnotOrder.ZZXX, 2, 3, epoch=1, logical_tick=3)
    logical_ex.CNOT(33, 22, 32, CnotOrder.ZZXX, 0, 2, epoch=1, logical_tick=4)
    logical_ex.Teleportation(33, 23, TeleportationType.XX, 0, epoch=0, logical_tick=5)
    logical_ex.tick()

    # T=7
    logical_ex.S_DAG_gate(11, 10, 3, epoch=0, logical_tick=4)
    logical_ex.CNOT(22, 11, 21, CnotOrder.ZZXX, 2, 3, epoch=2, logical_tick=3)
    logical_ex.CNOT(33, 22, 32, CnotOrder.ZZXX, 0, 2, epoch=2, logical_tick=4)
    logical_ex.Teleportation(33, 23, TeleportationType.XX, 0, epoch=1, logical_tick=5)
    logical_ex.tick()

    # T=8
    logical_ex.S_DAG_gate(11, 10, 3, epoch=1, logical_tick=4)
    logical_ex.CNOT(22, 10, 21, CnotOrder.ZZXX, 2, 3, epoch=3, logical_tick=3)
    logical_ex.CNOT(33, 22, 32, CnotOrder.ZZXX, 0, 2, epoch=3, logical_tick=4)
    logical_ex.Teleportation(33, 23, TeleportationType.XX, logical_qubit=0, epoch=2, logical_tick=5)
    logical_ex.tick()

    # T=9
    logical_ex.Teleportation(22, 32, TeleportationType.XX, 2, epoch=0, logical_tick=5)
    logical_ex.S_DAG_gate(11, 10, 3, epoch=2, logical_tick=4)
    logical_ex.tick()

    # T=10
    logical_ex.Teleportation(22, 32, TeleportationType.XX, 2, epoch=1, logical_tick=5)
    logical_ex.Teleportation(32, 33, TeleportationType.ZZ, 2, epoch=0, logical_tick=6)
    logical_ex.CNOT(22, 11, 21, CnotOrder.XXZZ, 0, 3, epoch=0, logical_tick=7)
    logical_ex.tick()

    # T=11
    logical_ex.S_DAG_gate(11, 10, 3, epoch=0, logical_tick=8)
    logical_ex.Teleportation(32, 33, TeleportationType.ZZ, 2, epoch=1, logical_tick=6)
    logical_ex.CNOT(22, 11, 21, CnotOrder.XXZZ, 0, 3, epoch=1, logical_tick=7)
    logical_ex.Teleportation(22, 32, TeleportationType.XX, 2, epoch=2, logical_tick=5)
    logical_ex.tick()

    # T=12
    logical_ex.S_DAG_gate(11, 10, 3, epoch=1, logical_tick=8)
    logical_ex.Teleportation(23, 22, TeleportationType.ZZ, 0, epoch=0, logical_tick=6)
    logical_ex.Teleportation(32, 33, TeleportationType.ZZ, 2, epoch=2, logical_tick=6)
    logical_ex.tick()

    # T=13
    logical_ex.S_DAG_gate(33, 32, 2, epoch=0, logical_tick=7)
    logical_ex.Teleportation(23, 22, TeleportationType.ZZ, 0, epoch=1, logical_tick=6)
    logical_ex.S_DAG_gate(11, 10, 3, epoch=2, logical_tick=8)
    logical_ex.tick()

    # T=14
    logical_ex.S_DAG_gate(33, 32, 2, epoch=1, logical_tick=7)
    logical_ex.CNOT(22, 11, 21, CnotOrder.XXZZ, 0, 3, epoch=2, logical_tick=7)
    logical_ex.Teleportation(23, 22, TeleportationType.ZZ, 0, epoch=2, logical_tick=6)
    logical_ex.tick()

    # T=15
    logical_ex.S_DAG_Rot_gate(11, 1, 3, epoch=0, logical_tick=9)
    logical_ex.CNOT(22, 33, 23, CnotOrder.ZZXX, 0, 2, epoch=0, logical_tick=8)
    logical_ex.logical_init(13, InitialState.X, 1)
    logical_ex.CNOT(13, 22, 12, CnotOrder.ZZXX, 1, 0, epoch=0, logical_tick=9)
    logical_ex.CNOT(22, 11, 21, CnotOrder.XXZZ, 0, 3, epoch=3, logical_tick=7)
    logical_ex.S_DAG_gate(33, 32, 2, epoch=2, logical_tick=7)
    logical_ex.tick()

    # T=16
    logical_ex.logical_init(31, InitialState.S_DAG, 4)
    logical_ex.CNOT(22, 33, 23, CnotOrder.ZZXX, 0, 2, epoch=1, logical_tick=8)
    logical_ex.CNOT(22, 31, 21, CnotOrder.XXZZ, 0, 4, epoch=0, logical_tick=10)
    logical_ex.CNOT(13, 22, 12, CnotOrder.ZZXX, 1, 0, epoch=1, logical_tick=9)
    logical_ex.S_DAG_Rot_gate(11, 1, 3, epoch=1, logical_tick=9)
    logical_ex.Teleportation(13, 3, TeleportationType.XX, 1, epoch=0, logical_tick=10)
    logical_ex.tick()

    # T=17
    logical_ex.S_gate(33, 32, 2, epoch=0, logical_tick=9)
    logical_ex.CNOT(22, 33, 23, CnotOrder.ZZXX, 0, 2, epoch=2, logical_tick=8)
    logical_ex.CNOT(22, 31, 21, CnotOrder.XXZZ, 0, 4, epoch=1, logical_tick=10)
    logical_ex.CNOT(13, 22, 12, CnotOrder.ZZXX, 1, 0, epoch=2, logical_tick=9)
    logical_ex.Teleportation(13, 3, TeleportationType.XX, 1, epoch=1, logical_tick=10)
    logical_ex.S_gate(11, 10, 3, epoch=0, logical_tick=10)
    logical_ex.S_DAG_Rot_gate(11, 1, 3, epoch=2, logical_tick=9)
    logical_ex.tick()

    # T=18
    logical_ex.S_gate(33, 32, 2, epoch=1, logical_tick=9)
    logical_ex.CNOT(22, 31, 21, CnotOrder.XXZZ, 0, 4, epoch=2, logical_tick=10)
    logical_ex.S_gate(11, 10, 3, epoch=1, logical_tick=10)
    logical_ex.CNOT(22, 33, 23, CnotOrder.ZZXX, 0, 2, epoch=3, logical_tick=8)
    logical_ex.CNOT(13, 22, 12, CnotOrder.ZZXX, 1, 0, epoch=3, logical_tick=9)
    logical_ex.Teleportation(13, 3, TeleportationType.XX, 1, epoch=2, logical_tick=10)
    logical_ex.tick()

    # T=19
    logical_ex.S_gate(22, 23, 0, epoch=0, logical_tick=11)
    logical_ex.Teleportation(11, 12, TeleportationType.ZZ, 3, epoch=0, logical_tick=11)
    logical_ex.S_gate(33, 32, 2, epoch=2, logical_tick=9)
    logical_ex.CNOT(22, 31, 21, CnotOrder.XXZZ, 0, 4, epoch=3, logical_tick=10)
    logical_ex.S_gate(11, 10, 3, epoch=2, logical_tick=10)
    logical_ex.tick()

    # T=20
    logical_ex.S_gate(31, 32, 4, epoch=0, logical_tick=11)
    logical_ex.Teleportation(11, 12, TeleportationType.ZZ, 3, epoch=1, logical_tick=11)
    logical_ex.S_gate(22, 23, 0, epoch=1, logical_tick=11)
    logical_ex.tick()

    # T=21
    logical_ex.S_gate(31, 32, 4, epoch=1, logical_tick=11)
    logical_ex.S_gate(22, 23, 0, epoch=2, logical_tick=11)
    logical_ex.Teleportation(11, 12, TeleportationType.ZZ, 3, epoch=2, logical_tick=11)
    logical_ex.tick()

    # T=22
    logical_ex.LongRangeCNOT([12, 11, 21, 31], LongRangeCnot.XXZZXX, 3, 4, epoch=0, logical_tick=12)
    logical_ex.S_gate(31, 32, 4, epoch=2, logical_tick=11)
    logical_ex.tick()

    # T=23
    logical_ex.S_DAG_gate(31, 32, 4, epoch=0, logical_tick=13)
    logical_ex.LongRangeCNOT([12, 11, 21, 31], LongRangeCnot.XXZZXX, 3, 4, epoch=1, logical_tick=12)
    logical_ex.Teleportation(12, 13, TeleportationType.ZZ, 3, epoch=0, logical_tick=13)
    logical_ex.CNOT(22, 13, 23, CnotOrder.ZZXX, 0, 3, epoch=0, logical_tick=14)
    logical_ex.tick()

    # T=24
    logical_ex.S_DAG_gate(31, 32, 4, epoch=1, logical_tick=13)
    logical_ex.Teleportation(33, 43, TeleportationType.XX, 2, epoch=0, logical_tick=10)
    logical_ex.CNOT(22, 13, 23, CnotOrder.ZZXX, 0, 3, epoch=1, logical_tick=14)
    logical_ex.LongRangeCNOT([12, 11, 21, 31], LongRangeCnot.XXZZXX, 3, 4, epoch=2, logical_tick=12)
    logical_ex.Teleportation(12, 13, TeleportationType.ZZ, 3, epoch=1, logical_tick=13)
    logical_ex.tick()

    # T=25
    logical_ex.Teleportation(33, 43, TeleportationType.XX, 2, epoch=1, logical_tick=10)
    logical_ex.CNOT(22, 13, 23, CnotOrder.ZZXX, 0, 3, epoch=2, logical_tick=14)
    logical_ex.Teleportation(43, 42, TeleportationType.ZZ, 2, epoch=0, logical_tick=11)
    logical_ex.Teleportation(12, 13, TeleportationType.ZZ, 3, epoch=2, logical_tick=13)
    logical_ex.LongRangeCNOT([12, 11, 21, 31], LongRangeCnot.XXZZXX, 3, 4, epoch=3, logical_tick=12)
    logical_ex.S_DAG_gate(31, 32, 4, epoch=2, logical_tick=13)
    logical_ex.tick()

    # T=26
    logical_ex.S_DAG_gate(13, 12, 3, epoch=0, logical_tick=15)
    logical_ex.CNOT(22, 31, 21, CnotOrder.ZZXX, 0, 4, epoch=0, logical_tick=17)

    logical_ex.Teleportation(43, 42, TeleportationType.ZZ, 2, epoch=1, logical_tick=11)
    logical_ex.CNOT(22, 13, 23, CnotOrder.ZZXX, 0, 3, epoch=3, logical_tick=14)
    logical_ex.Teleportation(33, 43, TeleportationType.XX, 2, epoch=2, logical_tick=10)
    logical_ex.tick()

    # T=27
    logical_ex.S_DAG_gate(13, 12, 3, epoch=1, logical_tick=15)
    logical_ex.CNOT(22, 31, 21, CnotOrder.ZZXX, 0, 4, epoch=1, logical_tick=17)
    logical_ex.CNOT(22, 13, 23, CnotOrder.ZZXX, 0, 3, epoch=0, logical_tick=16)
    logical_ex.Teleportation(43, 42, TeleportationType.ZZ, 2, epoch=2, logical_tick=11)
    logical_ex.tick()

    # T=28
    logical_ex.S_gate(31, 32, 4, epoch=0, logical_tick=18)
    logical_ex.CNOT(22, 31, 21, CnotOrder.ZZXX, 0, 4, epoch=2, logical_tick=17)
    logical_ex.CNOT(22, 13, 23, CnotOrder.ZZXX, 0, 3, epoch=1, logical_tick=16)
    logical_ex.S_DAG_gate(13, 12, 3, epoch=2, logical_tick=15)
    logical_ex.tick()

    # T=29
    logical_ex.S_gate(31, 32, 4, epoch=1, logical_tick=18)
    logical_ex.S_DAG_Rot_gate(31, 41, 4, epoch=0, logical_tick=19)
    logical_ex.CNOT(22, 13, 23, CnotOrder.ZZXX, 0, 3, epoch=2, logical_tick=16)
    logical_ex.CNOT(22, 31, 21, CnotOrder.ZZXX, 0, 4, epoch=3, logical_tick=17)
    logical_ex.tick()

    # T=30
    logical_ex.S_gate(31, 30, 4, epoch=0, logical_tick=20)
    logical_ex.CNOT(22, 31, 21, CnotOrder.ZZXX, 0, 4, epoch=0, logical_tick=21)
    logical_ex.S_DAG_Rot_gate(31, 41, 4, epoch=1, logical_tick=19)
    logical_ex.CNOT(22, 13, 23, CnotOrder.ZZXX, 0, 3, epoch=3, logical_tick=16)
    logical_ex.logical_measure(13, MeasurementBasis.X, 3, logical_tick=17)
    logical_ex.S_gate(31, 32, 4, epoch=2, logical_tick=18)
    logical_ex.tick()

    # T=31
    logical_ex.Teleportation(3, 13, TeleportationType.XX, 1, epoch=0, logical_tick=15)
    logical_ex.CNOT(22, 31, 21, CnotOrder.ZZXX, 0, 4, epoch=1, logical_tick=21)
    logical_ex.S_gate(31, 30, 4, epoch=1, logical_tick=20)
    logical_ex.S_DAG_Rot_gate(31, 41, 4, epoch=2, logical_tick=19)
    logical_ex.tick()

    # T=32
    logical_ex.CNOT(42, 31, 41, CnotOrder.XXZZ, 2, 4, epoch=0, logical_tick=22)
    logical_ex.Teleportation(3, 13, TeleportationType.XX, 1, epoch=1, logical_tick=15)
    logical_ex.CNOT(22, 31, 21, CnotOrder.ZZXX, 0, 4, epoch=2, logical_tick=21)
    logical_ex.S_gate(31, 30, 4, epoch=2, logical_tick=20)
    logical_ex.tick()

    # T=33
    logical_ex.CNOT(42, 31, 41, CnotOrder.XXZZ, 2, 4, epoch=1, logical_tick=22)
    logical_ex.CNOT(22, 31, 21, CnotOrder.ZZXX, 0, 4, epoch=3, logical_tick=21)
    logical_ex.Teleportation(3, 13, TeleportationType.XX, 1, epoch=2, logical_tick=15)
    logical_ex.logical_measure(22, MeasurementBasis.X, 0, logical_tick=22)
    logical_ex.tick()

    # T=34
    logical_ex.Teleportation(13, 23, TeleportationType.XX, 1, epoch=0, logical_tick=16)
    logical_ex.Teleportation(42, 32, TeleportationType.XX, 2, epoch=0, logical_tick=23)
    logical_ex.CNOT(42, 31, 41, CnotOrder.XXZZ, 2, 4, epoch=2, logical_tick=22)
    logical_ex.logical_measure(31, MeasurementBasis.X, 4, logical_tick=23)
    logical_ex.tick()

    # T=35
    logical_ex.CNOT(23, 32, 22, CnotOrder.ZZXX, 1, 2, epoch=0, logical_tick=24)
    logical_ex.Teleportation(13, 23, TeleportationType.XX, 1, epoch=1, logical_tick=16)
    logical_ex.Teleportation(42, 32, TeleportationType.XX, 2, epoch=1, logical_tick=23)
    logical_ex.CNOT(42, 31, 41, CnotOrder.XXZZ, 2, 4, epoch=3, logical_tick=22)
    logical_ex.tick()

    # T=36
    logical_ex.CNOT(23, 32, 22, CnotOrder.ZZXX, 1, 2, epoch=1, logical_tick=24)
    logical_ex.Teleportation(42, 32, TeleportationType.XX, 2, epoch=2, logical_tick=23)
    logical_ex.Teleportation(13, 23, TeleportationType.XX, 1, epoch=2, logical_tick=16)
    logical_ex.tick()

    # T=37
    logical_ex.CNOT(23, 32, 22, CnotOrder.ZZXX, 1, 2, epoch=2, logical_tick=24)
    logical_ex.logical_measure(23, MeasurementBasis.X, 1, logical_tick=25)
    logical_ex.tick()

    # T=38
    logical_ex.CNOT(23, 32, 22, CnotOrder.ZZXX, 1, 2, epoch=3, logical_tick=24)
    logical_ex.logical_measure(32, MeasurementBasis.X, 2, logical_tick=25)

    logical_ex.propagate_frames()
    logical_ex.add_obserable([logical_ex.logical_qubits[0], logical_ex.logical_qubits[4]],
                             [MeasurementBasis.X, MeasurementBasis.X], 0)
    logical_ex.add_obserable([logical_ex.logical_qubits[1], logical_ex.logical_qubits[2]],
                             [MeasurementBasis.X, MeasurementBasis.X], 1)
    logical_ex.add_obserable([logical_ex.logical_qubits[3]], [MeasurementBasis.X], 2)

    return logical_ex

##
def build_memory():
    logical_ex = Logical_Experiment(1, 1, logical_qubits=1)
    logical_ex.logical_init(0, InitialState.Z, 0)
    logical_ex.logical_measure(0, MeasurementBasis.Z, 0,logical_tick=1)
    logical_ex.add_obserable([logical_ex.logical_qubits[0]], [MeasurementBasis.Z], 0)
    return logical_ex

def build_memory_18d3():
    logical_ex = Logical_Experiment(2, 2, logical_qubits=2)
    logical_ex.logical_init(0, InitialState.Z, 0)
    logical_ex.tick()
    logical_ex.logical_init(11, InitialState.X, 1)
    logical_ex.tick()
    logical_ex.CNOT(0,11,1,CnotOrder.ZZXX,logical_control=0,logical_target=1,logical_tick=1,epoch=0)
    logical_ex.tick()
    logical_ex.CNOT(0, 11, 1, CnotOrder.ZZXX, logical_control=0, logical_target=1, logical_tick=1, epoch=1)
    logical_ex.tick()
    logical_ex.CNOT(0, 11, 1, CnotOrder.ZZXX, logical_control=0, logical_target=1, logical_tick=1, epoch=2)
    logical_ex.tick()
    logical_ex.CNOT(0, 11, 1, CnotOrder.ZZXX, logical_control=0, logical_target=1, logical_tick=1, epoch=3)
    logical_ex.tick()
    logical_ex.logical_measure(0, MeasurementBasis.Z, 0,logical_tick=2)
    logical_ex.tick()
    logical_ex.logical_measure(11, MeasurementBasis.X,1,logical_tick=3)

    logical_ex.add_obserable([logical_ex.logical_qubits[0]], [MeasurementBasis.Z], 0)
    logical_ex.add_obserable([logical_ex.logical_qubits[1]], [MeasurementBasis.X], 1)
    return logical_ex

def non_FT_init():
    logical_ex = Logical_Experiment(1, 1, logical_qubits=1)
    logical_ex.logical_init(0, InitialState.S, 0)
    logical_ex.logical_measure(0, MeasurementBasis.Y, 0, logical_tick=1)
    logical_ex.add_obserable([logical_ex.logical_qubits[0]], [MeasurementBasis.Y], 0)
    return logical_ex

def non_FT_circ():
    logical_ex = Logical_Experiment(2, 2, logical_qubits=1)
    logical_ex.logical_init(0, InitialState.S, 0)
    logical_ex.S_gate(0, 1, 0, epoch=0, logical_tick=1)
    logical_ex.tick()
    logical_ex.S_gate(0, 1, 0, epoch=1, logical_tick=1)
    logical_ex.tick()
    logical_ex.S_gate(0, 1, 0, epoch=2, logical_tick=1)
    logical_ex.logical_measure(0, MeasurementBasis.X, 0, logical_tick=2)

    logical_ex.propagate_frames()
    logical_ex.add_obserable([logical_ex.logical_qubits[0]], [MeasurementBasis.X], 0)
    return logical_ex
