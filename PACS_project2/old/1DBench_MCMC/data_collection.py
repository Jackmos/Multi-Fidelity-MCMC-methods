import numpy as np
from numba import jit

# Basic case
def create_basic_functions():
    @jit
    def modified_highfid(x, delta):
        period = 5.54
        phase_within_period = np.mod(x + delta / 6, period)
        return (6. * phase_within_period / 5 - 2.) ** 2 * np.sin(12. * phase_within_period / 5 - 4.)
    
    @jit
    def modified_lowfid(x, delta):
        period2 = 5.96
        phase_within_period = np.mod(x + 0.2 + delta / 6, period2)
        return 0.5 * (6. * phase_within_period / 5 - 2.) ** 2 * np.sin(12. * phase_within_period / 5 - 4.) + 10 * (phase_within_period / 5 - 0.5) + 5.

    return modified_highfid, modified_lowfid

# Discontinuous case
def create_discontinuous_functions():
    @jit
    def modified_highfid(x, delta):
        return (2*modified_lowfid(x,delta)- 20*x/5.+20)*(x/5.<0.5) + (4+2*modified_lowfid(x,delta)- 20*x/5.+20+delta)*(x/5.>0.5)  
        
    @jit
    def modified_lowfid(x, delta):
        return (0.5*(6.*x/5.-2.)**2*np.sin(12.*x/5.-4)+10.*(x/5-0.5)-5.)*(x<2.5) + (3+0.5*(6.*x/5.-2)**2*np.sin(12.*x/5.-4)+10*(x/5.-0.5)-5.+delta)*(x>2.5)

    return modified_highfid, modified_lowfid

# Oscillatory case
def create_oscillatory_functions():
    @jit
    def modified_highfid(x, delta):
        return (x/5-np.sqrt(2))*modified_lowfid(x, delta)**2

    @jit
    def modified_lowfid(x, delta):
        return np.sin(delta*x) 

    return modified_highfid, modified_lowfid



def get_parameters(example):
    if example == "Basic":
        modified_highfid, modified_lowfid = create_basic_functions()

        # highfid = lambda x,delta: (6.*(x+delta/6)/5-2.)**2 * np.sin(12.*(x+delta/6)/5-4.)
        # lowfid = lambda x, delta: 0.5*highfid(x, delta) + 10*((x-delta/6)/5-0.5) + 5.
        

        Nhf = 50
        Nlf = 100
        NepoLF = 2000
        NepoHF = 2000
        deltas=np.array([0,10,20])

    elif example == "Discontinuous":
        modified_highfid, modified_lowfid = create_discontinuous_functions()


                
        Nhf = 16
        Nlf = 40
        NepoLF = 2000
        NepoHF = 5200
        deltas=np.linspace(0.,15.,5)

    elif example == "Oscillatory":
        modified_highfid, modified_lowfid = create_oscillatory_functions()
        
        lowfid = lambda x: np.sin(8*np.pi*x/5)
        highfid = lambda x: (x/5-np.sqrt(2))*lowfid(x)**2
        
        Nhf = 15
        Nlf = 64
        NepoLF = 1000
        NepoHF = 3000
        deltas=np.linspace(2/5,8/5,4)*np.pi
    else:
        raise ValueError(f"Unsupported example type: {example}")
    
    xhf = np.linspace(0,5,Nhf)
    xlf = np.linspace(0,5,Nlf)
    #x_test = np.linspace(0,5,1000)

    datahf= np.array(np.meshgrid(xhf,deltas)).T.reshape(-1, 2)
    ord_index = np.lexsort((datahf[:, 0], datahf[:, 1]))
    datahf = datahf[ord_index]
    Yhf = modified_highfid(datahf[:,0],datahf[:,1])
    datalf= np.array(np.meshgrid(xlf,deltas)).T.reshape(-1, 2)
    ord_index = np.lexsort((datalf[:, 0], datalf[:, 1]))
    datalf = datalf[ord_index]
    Ylf = modified_lowfid(datalf[:,0],datalf[:,1])


    return {
        "modified_highfid": modified_highfid,
        "modified_lowfid": modified_lowfid,
        # "highfid": highfid,
        # "lowfid": lowfid,
        "Nhf": Nhf,
        "Nlf": Nlf,
        "xhf": xhf,
        "xlf": xlf,
        "Yhf": Yhf,
        "Ylf": Ylf,
        "NepoLF": NepoLF,
        "NepoHF": NepoHF,
        "deltas": deltas
    }




# potrebbero andare in utils
def process_data(datahf, parameters, t_eval, Yhf):
    """
    Find the elements of a dataset nearest to the ones given
    """
    indices = np.where(datahf[:,1] == parameters)[0]
    if len(indices)==0:
        raise ValueError(f" No observations related to parameter: {parameters}")
    
    datahf_values = datahf[indices, 0]
    datahf_values = datahf_values[:, np.newaxis]
    t_eval_values = t_eval[:, 0]  # Simplify t_eval to a 1D array if not already
    t_eval_values = t_eval_values[np.newaxis, :]
    differences = np.abs(datahf_values - t_eval_values)
    closest_indices = np.argmin(differences, axis=0)
    nearest_x = datahf[indices[closest_indices], 0]  # x values closest to each t_eval
    y_obs = Yhf[indices[closest_indices]]  # Corresponding y values from Yhf

    return nearest_x, y_obs



def run_simulation(datahf, mean_prior, cov_prior, Yhf, sigma_noise, n_data, parameters, sigma, rmwh_scaling, rwmh_cov, rwmh_adaptive, iterations, burnin, n_chains, final_model,algo):
    
    error = np.zeros((len(sigma_noise), len(n_data), sigma.shape[0], rmwh_scaling.shape[0]))
    estimate_MF=np.zeros((len(sigma_noise), len(n_data), sigma.shape[0], rmwh_scaling.shape[0]))
    for i,noise in enumerate(sigma_noise):

        print(f"Noise: {noise}")

        for k,n in enumerate(n_data):
            print(f"Number of data: {n}")
            for t,s in enumerate(sigma):

                for j,r in enumerate(rmwh_scaling):
                    
                    t_eval = np.linspace(0., 5., n).reshape(-1,1)
                    
                    nearest_x, y_obs = process_data(datahf, parameters, t_eval, Yhf)
                    
                    estimate_MF[i,k,t,j],error[i,k,t,j] = final_model.param_inverse(mean_prior, t_eval, cov_prior=cov_prior, rmwh_scaling=r, cov_noise=noise, cov_likelihood=s**2 * np.eye(t_eval.shape[0]), y_obs=y_obs, x_real=parameters, number_chains=n_chains, N=iterations, burn_in=burnin, diagnostic=True, rwmh_cov=rwmh_cov, rwmh_adaptive=rwmh_adaptive, algo=algo)
                    
        smallest_index = np.unravel_index(np.argmin(error), error.shape)
        print(f"The best estimate is given by: sigma_noise={sigma_noise[smallest_index[0]]}, number of data= {n_data[smallest_index[1]]}, sigma={sigma[smallest_index[2]]}, rmwh_scaling={rmwh_scaling[smallest_index[3]]}")

    return estimate_MF[smallest_index],error



    