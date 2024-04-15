# Script adapted from Palacio-Rodriguez et al. at https://github.com/kpalaciorodr/KTR/tree/master from J. Phys. Chem. Lett. 2022, 13, 32, 7490-7496.

import numpy as np
from sys import argv
import random
import glob
from scipy import interpolate, optimize, integrate
from scipy.stats import ks_1samp, ks_2samp
from scipy.stats import gamma as gamma_func
import warnings
import multiprocessing as mp
from functools import partial
import pandas as pd

warnings.filterwarnings('ignore')

def bootstrap(sample,func,nresamples,double=False):
    stat = []
    stat2 = []
    for i in range(nresamples):
        resample = random.choices(sample, k=len(sample))
        if double:
            a, b = func(resample)
            stat.append(a)
            stat2.append(b)
        else:
            stat.append(func(resample))
    if double:
        return np.std(stat), np.std(stat2)
    else:
        return np.std(stat)

# Negating splines to maximize them with SciPy minimize
def neg_spline(x, spline):
    return -1*spline(x)

# Infrequent Metadynamics Tiwary Estimator
def iMetaD_invMRT(times, event, rescale=False, acc=None):
    if rescale and acc is not None:
        times *= acc
    elif rescale:
        print("You need to provide the acceleration factors if rescale is True. If your times are already rescaled, set rescale to False.")
        return None
    elif acc is not None:
        print("Not using the provided acceleration factors. If you want to rescale the times, set rescale to True.")
    return event.sum() / np.sum(times[event])

# Infrequent Metadynamics CDF Fit
def iMetaD_CDF(t, k):
    return 1 - np.exp(-k * t)

def iMetaD_FitCDF(times, event, k_guess):
    M = event.sum() # Number of transitions
    N = len(event) # Number of simulations
    
    # Construct Empirical CDF
    ecdfx = np.sort(times[event])
    ecdfy = np.arange(1, M+1) / N

    # Fit Poisson distribution CDF to data
    return optimize.curve_fit(iMetaD_CDF, ecdfx, ecdfy, p0=k_guess)[0][0]

# Kramers' Time-dependent Rate method log likelihood function. (The logTrick uses the log-sum-exp trick to ideally increase precision for large exponents.)
def KTR_calculate_log_l(gamma, event, t, spline, cores=4, logTrick=False):

    p =  mp.Pool(cores)
    func = partial(KTR_calculate_cum_hazard, gamma, spline, logTrick)
    cum_hazard = np.array(p.map(func, t))
    p.close()

    log_hazard = KTR_calculate_log_hazard(gamma, t, spline)

    mean_t = cum_hazard.sum() / event.sum()
    log_l = -event.sum() * np.log(mean_t) + log_hazard[event].sum() - (1 / mean_t) * cum_hazard.sum()

    return -log_l

def KTR_calculate_cum_hazard(gamma, spline, logTrick, t):
    dt=1.
    if logTrick:
        max_spline = optimize.minimize_scalar(neg_spline, args=(spline)).x
        t_points = np.arange(0,t,dt)
        return 0.5*dt*(1 + np.exp(gamma*spline(t)) + 2*np.exp(gamma*max_spline + np.log(np.exp(gamma*spline(t_points[1:]) - gamma*max_spline).sum())))
    else:
        int_Veff = integrate.quad(lambda x: np.exp(gamma * spline(x)), 0, t)[0]
        return int_Veff

def KTR_calculate_log_hazard(gamma, t, spline):

    Veff = spline(t)
    return gamma * Veff

def KTR_CDF(t, k0, gamma, spline, cores=4, logTrick=False):
    p = mp.Pool(4)
    func = partial(KTR_calculate_cum_hazard, gamma, spline, logTrick)
    cum_hazard = np.array(p.map(func, t))
    p.close()
    return 1 - np.exp(-k0 * cum_hazard)

def avg_max_bias(maxbias, data, colvars_count, colvars_maxrow_count, beta):
    vmb_data = np.empty((colvars_count, colvars_maxrow_count))
    vmb_data.fill(np.nan)
    ix_col = None
    def fill_vmb_data(colvar_index):
        vmb_column_data = maxbias[colvar_index]
        diff_rows = colvars_maxrow_count - vmb_column_data.shape[0]
        if 0 < diff_rows:
            fill_diff = np.empty(diff_rows)
            fill_diff.fill(np.nan)
            vmb_column_data = np.hstack((vmb_column_data, fill_diff))
        vmb_data[colvar_index,:] = vmb_column_data
        return data[colvar_index][:,0] if data[colvar_index][:,0].shape[0] == colvars_maxrow_count else None

    for i in range(colvars_count):
        i_ix_col = fill_vmb_data(i)
        if None is not i_ix_col:
            ix_col = i_ix_col

    masked_vmb = np.ma.masked_array(vmb_data, np.isnan(vmb_data))
    vmb_average = np.ma.average(masked_vmb.T, axis=1)
    vmb_average = np.vstack((ix_col, vmb_average)).T
    vmb_average[:,1] *= beta

    return vmb_average

def KTR_MLE_rate(vmb_average, t, event, gamma_bounds, cores, logTrick):
    unique_T = vmb_average[:,0]
    unique_V = vmb_average[:,1]
    spline = interpolate.UnivariateSpline(unique_T, unique_V, s=0, ext=3)
    opt = optimize.minimize_scalar(KTR_calculate_log_l, bounds=gamma_bounds, method='bounded', args=(event, t, spline, cores, logTrick))
    gamma = opt.x

    p =  mp.Pool(cores)
    func = partial(KTR_calculate_cum_hazard, gamma, spline, logTrick)
    cum_hazard = np.array(p.map(func, t))
    p.close()

    mean_t = cum_hazard.sum() / event.sum()
    return np.array([1/mean_t, gamma]), spline

def KTR_CDF_rate(vmb_average, t, gamma_bounds, event, cores, logTrick, k_guess):
    
    # 2-parameter CDF fitting for gamma and k0
    counts = np.sort(t[event])
    ecdf = np.arange(1, event.sum()+1) / len(event)
    ecdf_data = np.column_stack((counts, ecdf))

    unique_T = vmb_average[:,0]
    unique_V = vmb_average[:,1]
    spline = interpolate.UnivariateSpline(unique_T, unique_V, s=0, ext=3)

    def KTR_CDF_simple(t,k0,gamma):
        return KTR_CDF(t,k0,gamma,spline,logTrick=logTrick)
    cdf_result = optimize.curve_fit(KTR_CDF_simple, ecdf_data[:,0], ecdf_data[:,1], p0=k_guess, bounds=([-np.inf,gamma_bounds[0]],[np.inf,gamma_bounds[1]]))
    return cdf_result[0], spline

def EATR_calculate_avg_acc(gamma, v_data, beta, ix_col, logTrick=False):
    if logTrick:
        simmax_v = np.nanmax(v_data, axis=0)
        masked_acc = np.ma.masked_array(np.exp(beta * gamma * (v_data - simmax_v)), np.isnan(v_data))
        acc_average = np.ma.average(masked_acc.T, axis=1)
        acc_average = beta*gamma*simmax_v + np.log(acc_average)
        acc_average = np.vstack((ix_col, acc_average)).T
        spline = interpolate.UnivariateSpline(acc_average[:,0], acc_average[:,1], s=0, ext=3)
        return spline
    else:
        masked_acc = np.ma.masked_array(np.exp(beta * gamma * v_data), np.isnan(v_data))
        acc_average = np.log(np.ma.average(masked_acc.T, axis=1))
        acc_average = np.vstack((ix_col, acc_average)).T
        spline = interpolate.UnivariateSpline(acc_average[:,0], acc_average[:,1], s=0, ext=3)
        return spline

def EATR_calculate_log_l(gamma, event, t, spline, cores=4, logTrick=False):

    p =  mp.Pool(cores)
    func = None
    func = partial(EATR_calculate_cum_hazard, gamma, spline, logTrick)
    cum_hazard = np.array(p.map(func, t))
    p.close()

    log_hazard = EATR_calculate_log_hazard(gamma, t, spline)
    #print(f"cum_hazard: {cum_hazard}")
    #print(f"log_hazard: {log_hazard}")

    mean_t = cum_hazard.sum() / event.sum()
    log_l = -event.sum() * np.log(mean_t) + log_hazard[event].sum() - (1 / mean_t) * cum_hazard.sum()

    return -log_l

def EATR_calculate_cum_hazard(gamma, spline, logTrick, t):
    dt=1.
    if logTrick:
        max_spline = optimize.minimize_scalar(neg_spline, args=(spline)).x
        t_points = np.arange(0,t,dt)
        return 0.5*dt*(1 + np.exp(spline(t)) + 2*np.exp(max_spline + np.log(np.exp(spline(t_points[1:]) - max_spline).sum())))
    else:
        int_Veff = integrate.quad(lambda x: np.exp(spline(x)), 0, t)[0]
        return int_Veff

def EATR_calculate_log_hazard(gamma, t, spline):

    Veff = spline(t)
    return Veff

def EATR_CDF(t, k0, gamma, spline, cores, logTrick=False):

    p =  mp.Pool(cores)
    func = partial(EATR_calculate_cum_hazard, gamma, spline, logTrick)
    cum_hazard = np.array(p.map(func, t))
    p.close()
    return 1 - np.exp(-cum_hazard * k0)

def inst_bias(data, colvars_count, colvars_maxrow_count, beta, biascol):
    v_data = np.empty((colvars_count, colvars_maxrow_count))
    v_data.fill(np.nan)
    ix_col = None
    def fill_v_data(colvar_index):
        v_column_data = data[colvar_index][:,biascol] #for Vinst, consider making a copy for VMB
        diff_rows = colvars_maxrow_count - v_column_data.shape[0]
        if 0 < diff_rows:
            fill_diff = np.empty(diff_rows)
            fill_diff.fill(np.nan)
            v_column_data = np.hstack((v_column_data, fill_diff))
        v_data[colvar_index,:] = v_column_data
        return data[colvar_index][:,0] if data[colvar_index][:,0].shape[0] == colvars_maxrow_count else None

    for i in range(colvars_count):
        i_ix_col = fill_v_data(i)
        if None is not i_ix_col:
            ix_col = i_ix_col

    return v_data, ix_col

def EATR_MLE_rate(v_data, t, event, gamma_bounds, beta, ix_col, cores, logTrick=False):
    def log_l_aa(gamma, event, t):
        spline = EATR_calculate_avg_acc(gamma, v_data, beta, ix_col, logTrick=logTrick)
        return EATR_calculate_log_l(gamma, event, t, spline, cores=cores, logTrick=logTrick)

    opt = optimize.minimize_scalar(log_l_aa, bounds=gamma_bounds, method='bounded', args=(event, t))
    gamma = opt.x

    spline = EATR_calculate_avg_acc(gamma, v_data, beta, ix_col, logTrick=logTrick)
    p = mp.Pool(cores)
    func = partial(EATR_calculate_cum_hazard, gamma, spline, logTrick)
    cum_hazard = np.array(p.map(func, t))
    p.close()

    mean_t = cum_hazard.sum() / event.sum()

    return np.array([1/mean_t, gamma]), spline

def EATR_CDF_rate(v_data, t, event, gamma_bounds, beta, ix_col, cores, k_guess, logTrick=False):

    def tcdf(time, k0, gamma):
        spline = EATR_calculate_avg_acc(gamma, v_data, beta, ix_col, logTrick=logTrick)
        return EATR_CDF(time, k0, gamma, spline, cores, logTrick=logTrick)

    # 2-parameter CDF fitting for gamma and k0
    counts = np.sort(t[event])
    ecdf = np.arange(1, event.sum() + 1) / len(event)
    ecdf_data=np.column_stack((counts, ecdf))

    cdf_result = optimize.curve_fit(tcdf, ecdf_data[:,0], ecdf_data[:,1], p0=k_guess, bounds=([-np.inf,gamma_bounds[0]],[np.inf,gamma_bounds[1]]))
    spline = EATR_calculate_avg_acc(cdf_result[0][1], v_data, beta, ix_col, logTrick=logTrick)
    return cdf_result[0], spline

def rates(directory,runs,analyses,columns,beta,gamma_bounds,colvar_name,log_name,plog_len,cores,ks_ranges=False,boots=False,logTrick=False):

    results = {
            "iMetaD MLE k": None,
            "iMetaD MLE std k": None,
            "iMetaD CDF k": None,
            "iMetaD CDF std k": None,
            "iMetaD CDF KS klo": None,
            "iMetaD CDF KS khi": None,
            "KTR Vmb MLE k": None,
            "KTR Vmb MLE std k": None,
            "KTR Vmb MLE g": None,
            "KTR Vmb MLE std g": None,
            "KTR Vmb CDF k": None,
            "KTR Vmb CDF std k": None,
            "KTR Vmb CDF g": None,
            "KTR Vmb CDF std g": None,
            "KTR Vmb CDF KS klo": None,
            "KTR Vmb CDF KS khi": None,
            "KTR Vmb CDF KS glo": None,
            "KTR Vmb CDF KS ghi": None,
            "EATR MLE k": None,
            "EATR MLE std k": None,
            "EATR MLE g": None,
            "EATR MLE std g": None,
            "EATR CDF k": None,
            "EATR CDF std k": None,
            "EATR CDF g": None,
            "EATR CDF std g": None,
            "EATR CDF KS klo": None,
            "EATR CDF KS khi": None,
            "EATR CDF KS glo": None,
            "EATR CDF KS ghi": None
    }
    
    # Figure out what sections of the code need to be run
    actions_needed = {
            "avg_max": analyses["KTR Vmb MLE"] or analyses["KTR Vmb CDF"],
            "avg_acc": analyses["EATR MLE"] or analyses["EATR CDF"]
    }

    
    # Gather colvar, HILLS, and PLUMED log files
    print(f"{directory}:")
    colvars = []
    plogs = []
    for run in runs:
        colvars.append(f"{directory}/{run}/{colvar_name}")
        plogs.append(f"{directory}/{run}/{log_name}")
    
    # Load all colvar files
    colvars_count = len(colvars)
    colvars_maxrow_count = None
    
    data = [] # data[i][j,k] is column k of simulation i at the time of row j.
    final_times = np.zeros((colvars_count, 2)) # final_times[i,0] is simulation i's transition time while final_times[i,1] is the iMetaD rescaled time.
    i = 0
    for colvar in colvars:
        current_colvar = np.loadtxt(colvar)
        data.append(current_colvar)
        colvars_maxrow_count = data[-1].shape[0] if colvars_maxrow_count is None or colvars_maxrow_count < data[-1].shape[0] else colvars_maxrow_count
        final_times[i,:] = np.array([data[-1][-1][columns["time"]],data[-1][-1][columns["time"]] * data[-1][-1][columns["acc"]]]) # final_times[i,0] is simulation i's transition time while final_times[i,1] is the iMetaD rescaled time.
        i = i+1

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
    print(f"{M} out of {N} underbiased simulations transitioned.")

    t = final_times[:,0] # Simulation times
    
    ### Infrequent Metadynamics ###
    # from Tiwary and Parinello paper
    taus = final_times[:,1] # Rescaled times
    k_mle = iMetaD_invMRT(taus, event) # Maximum likelihood estimate of the rate, not assuming all simulations transitioned

    def tcdf(x, k):
        return 1 - np.exp(-k * x)

    def select_runs_iMetaD_MLE(run_idx):
        temp_taus = taus[run_idx]
        temp_event = event[run_idx]
        return np.log10(iMetaD_invMRT(temp_taus, temp_event))

    if analyses["iMetaD MLE"]:
        # Compute Kolmogorov-Smirnov Statistic
        size = np.int64(len(final_times[:,1])*5e4)
        rvs1 = gamma_func.rvs(1, scale=1/k_mle, size=size)
        ks_stat, p = ks_2samp(rvs1,taus[event])
        results["iMetaD MLE k"] = k_mle
        if boots:
            res = bootstrap(list(range(N)),select_runs_iMetaD_MLE,100)
            results["iMetaD MLE std k"] = res
            print(f"iMetaD MLE: logk = {np.log10(k_mle)} +/- {res}, KS: {ks_stat}, p = {p}")
        else:
            print(f"iMetaD MLE: k = {k_mle}, KS: {ks_stat}, p = {p}")


    # from Salvalaglio paper
    if analyses["iMetaD CDF"]:

        # Fit ECDF to theoretical CDF
        k = iMetaD_FitCDF(taus, event, k_mle)

        # Compute Kolmogorov-Smirnov Statistic
        size = np.int64(len(final_times[:,1])*5e4)
        rvs1 = gamma_func.rvs(1, scale=1/k, size=size)
        ks_stat, p = ks_2samp(rvs1,final_times[:,1]) # 2-sample test because that's what's commonly used, apparently.
        results["iMetaD CDF k"] = k

        if boots:
            def select_runs_iMetaD_CDF(run_idx):
                temp_taus = taus[run_idx]
                temp_event = event[run_idx]
                return np.log10(iMetaD_FitCDF(temp_taus, temp_event, k_mle))
            res = bootstrap(list(range(N)),select_runs_iMetaD_CDF,100)
            results["iMetaD CDF std k"] = res
            print(f"iMetaD CDF: logk = {np.log10(k)} +/- {res}, KS: {ks_stat}, p = {p}")
        else:
            print(f"iMetaD CDF: k = {k}, KS: {ks_stat}, p = {p}")

        if ks_ranges:
            
            good_fit = None
            bounds = []

            k_i = k
            p = 0.06
            while p > 0.05:
                k_i *= 10**(-0.02)
                rvs1 = gamma_func.rvs(1, scale=1/k_i, size=size)
                _, p = ks_2samp(rvs1,final_times[:,1])
                if p > 0.05:
                    good_fit = k_i
            if good_fit is not None:
                results["iMetaD CDF KS klo"] = good_fit
            else:
                results["iMetaD CDF KS klo"] = k

            good_fit = None
            k_i = k
            p = 0.06
            while p > 0.05:
                k_i *= 10**(0.02)
                rvs1 = gamma_func.rvs(1, scale=1/k_i, size=size)
                _, p = ks_2samp(rvs1,final_times[:,1])
                if p > 0.05:
                    good_fit = k_i
            if good_fit is not None:
                results["iMetaD CDF KS khi"] = good_fit
            else:
                results["iMetaD CDF KS khi"] = k

            print(f"iMetaD CDF Passes: k0: {results['iMetaD CDF KS klo']} to {results['iMetaD CDF KS khi']}")


    # Estimate maximum bias from colvar data and save to max_accumulate if we're using a max bias measure.
    max_accumulate = []
    if actions_needed["avg_max"]:
        if columns["max_bias"] is None: 
            def set_max_accumulate(dataset, maxset, numcol = columns["bias"]):
                max_value_found = None
                for i in range(dataset.shape[0]):
                    max_value_found = dataset[i][numcol] if max_value_found is None or max_value_found < dataset[i][numcol] else max_value_found
                    maxset[i] = max_value_found
        
            for i in range(N):
                max_accumulate.append(np.zeros_like(data[i][:,columns["bias"]]))
                set_max_accumulate(data[i],max_accumulate[i])
        else:
            for i in range(N):
                max_accumulate.append(data[i][:,columns["max_bias"]].copy())
        
        vmb_average = avg_max_bias(max_accumulate, data, colvars_count, colvars_maxrow_count, beta)

        # Finally calculate the rate
        mle_result, spline = KTR_MLE_rate(vmb_average, t, event, gamma_bounds, cores, logTrick)

        if analyses["KTR Vmb MLE"]:
            ks_stat, p = ks_1samp(final_times[:,0],KTR_CDF,args=(mle_result[0],mle_result[1],spline,logTrick)) # 1-sample test because the KTR CDF takes a while to sample, apparently.
            results["KTR Vmb MLE k"] = mle_result[0]
            results["KTR Vmb MLE g"] = mle_result[1]

            if boots:
                def select_runs_KTR_MLE(run_idx):
                    temp_t = t[run_idx]
                    temp_event = event[run_idx]
                    temp_maxbias = np.array(max_accumulate)[run_idx]
                    temp_data = np.array(data)[run_idx]
                    temp_cc = len(run_idx)
                    temp_cmc = np.max([len(c) for c in temp_data])

                    temp_vmb = avg_max_bias(temp_maxbias, temp_data, temp_cc, temp_cmc, beta)
                    result, _ = KTR_MLE_rate(temp_vmb, temp_t, temp_event, gamma_bounds, cores, logTrick)
                    return np.log10(result[0]), result[1]

                res_k, res_g = bootstrap(list(range(N)),select_runs_KTR_MLE,100,double=True)
                
                results["KTR Vmb MLE std k"] = res_k
                results["KTR Vmb MLE std g"] = res_g
                print(f"KTR Vmb MLE: logk = {np.log10(mle_result[0])} +/- {res_k}, gamma: {mle_result[1]} +/- {res_g}, KS: {ks_stat}, p = {p}")
            else:
                print(f"KTR Vmb MLE: k = {mle_result[0]}, gamma: {mle_result[1]}, KS: {ks_stat}, p = {p}")
        
        if analyses["KTR Vmb CDF"]:
            
            cdf_result, spline = KTR_CDF_rate(vmb_average, t, gamma_bounds, event, cores, logTrick, mle_result)
            ks_stat, p = ks_1samp(final_times[:,0],KTR_CDF,args=(cdf_result[0],cdf_result[1],spline,logTrick))
            results["KTR Vmb CDF k"] = cdf_result[0]
            results["KTR Vmb CDF g"] = cdf_result[1]

            if boots:
                def select_runs_KTR_CDF(run_idx):
                    temp_t = t[run_idx]
                    temp_event = event[run_idx]
                    temp_maxbias = np.array(max_accumulate)[run_idx]
                    temp_data = np.array(data)[run_idx]
                    temp_cc = len(run_idx)
                    temp_cmc = np.max([len(c) for c in temp_data])

                    temp_vmb = avg_max_bias(temp_maxbias, temp_data, temp_cc, temp_cmc, beta)
                    result, _ = KTR_CDF_rate(temp_vmb, temp_t, gamma_bounds, temp_event, cores, logTrick, mle_result)
                    return np.log10(result[0]), result[1]

                res_k, res_g = bootstrap(list(range(N)),select_runs_KTR_CDF,100,double=True)

                results["KTR Vmb CDF std k"] = res_k
                results["KTR Vmb CDF std g"] = res_g
                print(f"KTR Vmb CDF: logk = {np.log10(cdf_result[0])} +/- {res_k}, gamma: {cdf_result[1]} +/- {res_g}, KS: {ks_stat}, p = {p}")
            else:
                print(f"KTR Vmb CDF: {cdf_result[0]}, gamma: {cdf_result[1]}, KS: {ks_stat}, p = {p}")

            if ks_ranges:
                good_fit = None
                gamma_i = cdf_result[1]
                p = 0.06
                while p > 0.05 and gamma_i > gamma_bounds[0]:
                    gamma_i -= 0.02
                    cdf_result_i, spline = KTR_CDF_rate(vmb_average, t, (gamma_i-0.00000000001,gamma_i), event, cores, logTrick, (mle_result[0],gamma_i))
                    #cdf_result_i = optimize.curve_fit(KTR_CDF, ecdf_data[:,0], ecdf_data[:,1], p0=(mle_result[0],gamma_i), bounds=([-np.inf,gamma_i-0.00000000001],[np.inf,gamma_i]))
                    _, p = ks_1samp(final_times[:,0],KTR_CDF,args=(cdf_result_i[0],cdf_result_i[1],spline,logTrick))
                    if p > 0.05:
                        good_fit = cdf_result_i
                if good_fit is not None:
                    results["KTR Vmb CDF KS khi"] = good_fit[0]
                    results["KTR Vmb CDF KS glo"] = good_fit[1]
                else:
                    results["KTR Vmb CDF KS khi"] = cdf_result[0]
                    results["KTR Vmb CDF KS glo"] = cdf_result[1]
                
                good_fit = None
                gamma_i = cdf_result[1]
                p = 0.06
                while p > 0.05 and gamma_i < gamma_bounds[1]:
                    gamma_i += 0.02
                    cdf_result_i, spline = KTR_CDF_rate(vmb_average, t, (gamma_i-0.00000000001,gamma_i), event, cores, logTrick, (mle_result[0],gamma_i))
                    #cdf_result_i = optimize.curve_fit(KTR_CDF, ecdf_data[:,0], ecdf_data[:,1], p0=(mle_result[0],gamma_i), bounds=([-np.inf,gamma_i-0.00000000001],[np.inf,gamma_i]))
                    _, p = ks_1samp(final_times[:,0],KTR_CDF,args=(cdf_result_i[0],cdf_result_i[1],spline,logTrick))
                    if p > 0.05:
                        good_fit = cdf_result_i
                if good_fit is not None:
                    results["KTR Vmb CDF KS klo"] = good_fit[0]
                    results["KTR Vmb CDF KS ghi"] = good_fit[1]
                else:
                    results["KTR Vmb CDF KS klo"] = cdf_result[0]
                    results["KTR Vmb CDF KS ghi"] = cdf_result[1]

                print(f"KTR Vmb CDF Passes: gamma: {results['KTR Vmb CDF KS glo']} to {results['KTR Vmb CDF KS ghi']}, k0: {results['KTR Vmb CDF KS klo']} to {results['KTR Vmb CDF KS khi']}")

        
    ### EATR Method ###
    if actions_needed["avg_acc"]:

        v_data, ix_col = inst_bias(data, colvars_count, colvars_maxrow_count, beta, columns["bias"])

        mle_result, spline = EATR_MLE_rate(v_data, t, event, gamma_bounds, beta, ix_col, cores, logTrick=logTrick)

        def tcdf(time, k0, gamma):
            return EATR_CDF(time, k0, gamma, spline, cores, logTrick=logTrick)

        if analyses["EATR MLE"]:
            ks_stat, p = ks_1samp(final_times[:,0],EATR_CDF,args=(mle_result[0],mle_result[1],spline,cores,logTrick))
            results["EATR MLE k"] = mle_result[0]
            results["EATR MLE g"] = mle_result[1]
            if boots:
                def select_runs_EATR_MLE(run_idx):
                    temp_t = t[run_idx]
                    temp_event = event[run_idx]
                    temp_data = np.array(data)[run_idx]
                    temp_cc = len(run_idx)
                    temp_cmc = np.max([len(c) for c in temp_data])

                    temp_v, temp_ixcol = inst_bias(temp_data, temp_cc, temp_cmc, beta, columns["bias"])
                    result, _ = EATR_MLE_rate(temp_v, temp_t, temp_event, gamma_bounds, beta, temp_ixcol, cores, logTrick=logTrick)
                    return np.log10(result[0]), result[1]
                    
                res_k, res_g = bootstrap(list(range(N)),select_runs_EATR_MLE,100,double=True)

                results["EATR MLE std k"] = res_k
                results["EATR MLE std g"] = res_g
                print(f"EATR MLE: logk = {np.log10(mle_result[0])} +/- {res_k}, gamma: {mle_result[1]} +/- {res_g}, KS: {ks_stat}, p = {p}")
            else:
                print(f"EATR MLE: {mle_result[0]}, gamma: {mle_result[1]}, KS: {ks_stat}, p = {p}")

        if analyses["EATR CDF"]:

            cdf_result, spline = EATR_CDF_rate(v_data, t, event, gamma_bounds, beta, ix_col, cores, mle_result, logTrick=logTrick)

            ks_stat, p = ks_1samp(final_times[:,0],EATR_CDF,args=(cdf_result[0],cdf_result[1],spline,cores,logTrick))
            results["EATR CDF k"] = cdf_result[0]
            results["EATR CDF g"] = cdf_result[1]

            if boots:
                def select_runs_EATR_CDF(run_idx):
                    temp_t = t[run_idx]
                    temp_event = event[run_idx]
                    temp_data = np.array(data)[run_idx]
                    temp_cc = len(run_idx)
                    temp_cmc = np.max([len(c) for c in temp_data])

                    temp_v, temp_ixcol = inst_bias(temp_data, temp_cc, temp_cmc, beta, columns["bias"])
                    result, _ = EATR_CDF_rate(temp_v, temp_t, temp_event, gamma_bounds, beta, temp_ixcol, cores, mle_result, logTrick=logTrick) # Make function
                    return np.log10(result[0]), result[1]

                res_k, res_g = bootstrap(list(range(N)),select_runs_EATR_CDF,100,double=True)

                results["EATR CDF std k"] = res_k
                results["EATR CDF std g"] = res_g
                print(f"EATR CDF: logk = {np.log10(cdf_result[0])} +/- {res_k}, gamma: {cdf_result[1]} +/- {res_g}, KS: {ks_stat}, p = {p}")
            else:
                print(f"EATR CDF: {cdf_result[0]}, gamma: {cdf_result[1]}, KS: {ks_stat}, p = {p}")

            if ks_ranges:
                good_fit = None
                gamma_i = cdf_result[1]
                p = 0.06
                while p > 0.05 and gamma_i > gamma_bounds[0]:
                    gamma_i -= 0.02
                    cdf_result_i, spline = EATR_CDF_rate(v_data, t, event, (gamma_i-0.00000000001,gamma_i), beta, ix_col, cores, (mle_result[0],gamma_i), logTrick=logTrick)
                    #cdf_result_i = optimize.curve_fit(tcdf, ecdf_data[:,0], ecdf_data[:,1], p0=(mle_result[0],gamma_i), bounds=([-np.inf,gamma_i-0.00000000001],[np.inf,gamma_i]))
                    _, p = ks_1samp(final_times[:,0],tcdf,args=(cdf_result_i[0],cdf_result_i[1]))
                    if p > 0.05:
                        good_fit = cdf_result_i
                if good_fit is not None:
                    results["EATR CDF KS khi"] = good_fit[0]
                    results["EATR CDF KS glo"] = good_fit[1]
                else:
                    results["EATR CDF KS khi"] = cdf_result[0]
                    results["EATR CDF KS glo"] = cdf_result[1]
                
                good_fit = None
                gamma_i = cdf_result[1]
                p = 0.06
                while p > 0.05 and gamma_i < gamma_bounds[1]:
                    gamma_i += 0.02
                    cdf_result_i, spline = EATR_CDF_rate(v_data, t, event, (gamma_i-0.00000000001,gamma_i), beta, ix_col, cores, (mle_result[0],gamma_i), logTrick=logTrick)
                    #cdf_result_i = optimize.curve_fit(tcdf, ecdf_data[:,0], ecdf_data[:,1], p0=(mle_result[0],gamma_i), bounds=([-np.inf,gamma_i-0.00000000001],[np.inf,gamma_i]))
                    _, p = ks_1samp(final_times[:,0],tcdf,args=(cdf_result_i[0],cdf_result_i[1]))
                    if p > 0.05:
                        good_fit = cdf_result_i
                if good_fit is not None:
                    results["EATR CDF KS klo"] = good_fit[0]
                    results["EATR CDF KS ghi"] = good_fit[1]
                else:
                    results["EATR CDF KS klo"] = cdf_result[0]
                    results["EATR CDF KS ghi"] = cdf_result[1]

                print(f"EATR CDF KS Passes: gamma: {results['EATR CDF KS glo']} to {results['EATR CDF KS ghi']}, k0: {results['EATR CDF KS klo']} to {results['EATR CDF KS khi']}")
    
    return results
