##this code runs the factorization circuit simulations.
# one may change d_vec and p to explore different outcomes
from logical_level_circuit import build_factorization_circuit
from analyze_circuit import analyze, analyze_from_circ

p_vec = [0.001, 0.003]
d_vec = [3, 5, 7, 9]

build_circuit = False

if build_circuit:
    logical_ex = build_factorization_circuit()
    analyze(logical_ex, p_vec, d_vec, task_name='fact_circuit', post_selection=False)
    analyze(logical_ex, p_vec, d_vec, task_name='fact_circuit', post_selection=True)
else:
    analyze_from_circ(p_vec, d_vec, task_name='fact_circuit', post_selection=False)
    analyze_from_circ(p_vec, d_vec, task_name='fact_circuit', post_selection=True)
