import numpy as np
import os, sys
from scipy import optimize, stats, interpolate
sys.path.append(os.path.abspath('.'))
import rate_methods as RM
import tracemalloc

log_name = 'p.log'
colvar_name = 'opes.colvar'
runs = [f"run_{i+1}" for i in range(100)]
plog_len = 1
beta = 1. / (312.*0.008314)

def get_data(directory):
    print(f"{directory}")
    colvars = []
    plogs = []
    for run in runs:
        colvars.append(f"../../Data/Q_frac_native_contacts_opes/{directory}/{run}/{colvar_name}")
        plogs.append(f"../../Data/Q_frac_native_contacts_opes/{directory}/{run}/{log_name}")
    
    # Load all colvar files
    colvars_count = len(colvars)
    colvars_maxrow_count = None
    
    data = [] # data[i][j,k] is column k of simulation i at the time of row j.
    final_times = np.zeros((colvars_count, 2)) # final_times[i,0] is simulation i's transition time while final_times[i,1] is the iMetaD rescaled time.
    i = 0
    for colvar in colvars:
        current_colvar = np.loadtxt(colvar, usecols=[0,3])
        data.append(current_colvar)
        colvars_maxrow_count = data[-1].shape[0] if colvars_maxrow_count is None or colvars_maxrow_count < data[-1].shape[0] else colvars_maxrow_count
        final_times[i,:] = np.array([data[-1][-1][0],0])
        i = i+1

    #tracemalloc.stop()
    # Count transitions
    event = []
    for plog in plogs:
        with open(plog,'r') as f:
            if len(f.readlines()) > plog_len:
                event.append(True)
            else:
                event.append(False)
    event = np.array(event)
    M = event.sum() # Number of transitions
    N = len(event) # Total number of simulations

    return data, final_times, colvars_count, colvars_maxrow_count, event

barrs = [5,7,9,11,13,15]
avgs = []
VMBs = []
emp_rates = []

for barr in barrs:
    trans_times = np.loadtxt(f"../../Data/Q_frac_native_contacts_opes/qruns_barr{barr}/trans_times.dat", usecols=0)
    emp_rates.append(1. / np.mean(trans_times))
    data, final_times, colvars_count, colvars_maxrow_count, event = get_data(f"qruns_barr{barr}")

    max_accumulate = []
    def set_max_accumulate(dataset, maxset, numcol = 1):
        max_value_found = None
        for i in range(dataset.shape[0]):
            max_value_found = dataset[i][numcol] if max_value_found is None or max_value_found < dataset[i][numcol] else max_value_found
            maxset[i] = max_value_found

    for i in range(len(event)):
        max_accumulate.append(np.zeros_like(data[i][:,1]))
        set_max_accumulate(data[i],max_accumulate[i])

    vmb = RM.avg_max_bias(max_accumulate, data, colvars_count, colvars_maxrow_count, beta, bias_shift=barr)
    unique_T = vmb[:,0]
    unique_V = vmb[:,1]
    spline_KTR = interpolate.UnivariateSpline(unique_T, unique_V, s=0, ext=3)

    v_data, ix_col = RM.inst_bias(data, colvars_count, colvars_maxrow_count, beta, 1, bias_shift=barr)
    spline_EATR = RM.EATR_calculate_avg_acc(1., v_data, beta, ix_col)
    
    ts = np.linspace(0,np.max(final_times[:,0]),500)
    ys_EATR = spline_EATR(ts)
    ys_KTR = spline_KTR(ts)

    avgs.append(np.mean(ys_EATR))
    VMBs.append(np.mean(ys_KTR))

def KTR_TDR(vmb,k0,gamma):
    return np.log(k0) + gamma*vmb

def EATR_TDR(avg,k0,gamma):
    return np.log(k0) + gamma*avg

print(f"KTR: {optimize.curve_fit(KTR_TDR, VMBs, np.log(emp_rates), p0=(1e-6,1.0))[0]}")
print(f"EATR: {optimize.curve_fit(EATR_TDR, avgs, np.log(emp_rates), p0=(1e-6,1.0))[0]}")
print("VMB logacc emprate")
for i in range(len(barrs)):
    print(VMBs[i], avgs[i], emp_rates[i])
