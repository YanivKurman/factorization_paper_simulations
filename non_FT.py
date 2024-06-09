from logical_level_circuit import non_FT_circ
from analyze_circuit import analyze
import pickle
import numpy as np
import matplotlib.pyplot as plt

save = False
run_simulation=False
plot=True

##
p_vec=[0.001,0.003,0.005,0.008,0.01]
d_vec=[3,5,7,9]
d3_mat = np.load('matrices/d3_errors_matrix.npy')

if run_simulation:
    logical_ex=non_FT_circ()
    for post_selection in [True,False]:
        analyze(logical_ex,p_vec, d_vec, post_selection,f'S_gate_PS={post_selection}')

else:
    for post_selection in [True,False]:
        file_path = f'results\S_gate_PS={post_selection}.pkl'
        with open(file_path, 'rb') as file:
            # Load the object from the file
            data = pickle.load(file)
        if not post_selection:
            nFT_noPS_errors=np.empty((len(p_vec), len(d_vec)))
            for indx in range(len(data)):
                ind_p=p_vec.index(data[indx].json_metadata['p'])
                ind_d=d_vec.index(data[indx].json_metadata['d'])
                nFT_noPS_errors[ind_p,ind_d]=data[indx].errors/data[indx].shots
            nFT_init_noPS_errors=1-((1-nFT_noPS_errors)/(1-d3_mat)**4)**(1/2)
            if save:
                np.save('matrices/nFT_noPS_init.npy', nFT_init_noPS_errors)
            if plot:
                for index, row in enumerate(nFT_init_noPS_errors.T):
                    plt.scatter(p_vec, row, label=f'd= {d_vec[index]}')
                    # plt.plot(p_vec, row, label=f'd= {d_vec[index]}')

                plt.yscale('log')
                plt.xscale('log')
                plt.ylim(10 ** -3, 2 * 10 ** -1)
                plt.grid()
                plt.legend()
                plt.show()
        else:
            nFT_PS_errors = np.empty((len(p_vec), len(d_vec)))
            nFT_discards = np.empty((len(p_vec), len(d_vec)))
            for indx in range(len(data)):
                ind_p = p_vec.index(data[indx].json_metadata['p'])
                ind_d = d_vec.index(data[indx].json_metadata['d'])
                nFT_PS_errors[ind_p, ind_d] = data[indx].errors / (data[indx].shots - data[indx].discards)
                nFT_discards[ind_p, ind_d] = data[indx].discards / data[indx].shots
            nFT_init_PS_errors = 1 - ((1 - nFT_PS_errors) / (1 - d3_mat) ** 4) ** (1 / 2)

            if save:
                np.save('matrices/nFT_PS_init.npy', nFT_init_PS_errors)
            if plot:
                for index, row in enumerate(nFT_init_PS_errors.T):
                    plt.scatter(p_vec,row, label=f'd= {d_vec[index]} PS',marker='d')
                    # plt.plot(p_vec,row, label=f'd= {d_vec[index]}')
