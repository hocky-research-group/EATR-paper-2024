import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

trajC = pd.read_csv("../Data/unbiased_cvs.colvar", delim_whitespace=True, header=None, skiprows=1)
trajL = pd.read_csv("../Data/unbiased_ld1.colvar", delim_whitespace=True, header=None, skiprows=1)

PR, binsR, _ = plt.hist(trajC.iloc[:,3],bins=80,density=True)
PQ, binsQ, _ = plt.hist(trajC.iloc[:,1],bins=80,density=True)
PL, binsL, _ = plt.hist(trajL.iloc[:,1],bins=80,density=True)
PG, binsG, _ = plt.hist(trajC.iloc[:,4],bins=80,density=True)
PE, binsE, _ = plt.hist(trajC.iloc[:,2],bins=80,density=True)
PGE, Gedges, Eedges, _ = plt.hist2d(trajC.iloc[:,4],trajC.iloc[:,2],bins=80,density=True)

def getX(bins):
    x = []
    y = []
    for i in range(len(bins)-1):
        x.append(0.5*(bins[i]+bins[i+1]))
    return x

def getX_2D(xedges,yedges):
    x = []
    y = []
    for i in range(len(xedges)-1):
        x.append(0.5*(xedges[i]+xedges[i+1]))
        y.append(0.5*(yedges[i]+yedges[i+1]))
    return x, y

PMFR = -0.008314 * 312 * np.log(PR)
R = getX(binsR)

PMFQ = -0.008314 * 312 * np.log(PQ)
Q = getX(binsQ)

PMFL = -0.008314 * 312 * np.log(PL)
L = getX(binsL)

PMFG = -0.008314 * 312 * np.log(PG)
G = getX(binsG)

PMFE = -0.008314 * 312 * np.log(PE)
E = getX(binsE)

PMFGE = -0.008314 * 312 * np.log(PGE)
G2, E2 = getX_2D(Gedges, Eedges)




with open("../Results/PMF/pmf_R.dat", 'w') as f:
    f.write('#RMSD A\n')
    for i in range(len(PMFR)):
        f.write(str(R[i]) + " " + str(PMFR[i]) + "\n")

with open("../Results/PMF/pmf_Q.dat", 'w') as f:
    f.write('#Q A\n')
    for i in range(len(PMFQ)):
        f.write(str(Q[i]) + " " + str(PMFQ[i]) + "\n")

with open("../Results/PMF/pmf_L.dat", 'w') as f:
    f.write('#LD1 A\n')
    for i in range(len(PMFL)):
        f.write(str(L[i]) + " " + str(PMFL[i]) + "\n")

with open("../Results/PMF/pmf_G.dat", 'w') as f:
    f.write('#Rg A\n')
    for i in range(len(PMFG)):
        f.write(str(G[i]) + " " + str(PMFG[i]) + "\n")

with open("../Results/PMF/pmf_E.dat", 'w') as f:
    f.write('#Ree A\n')
    for i in range(len(PMFE)):
        f.write(str(E[i]) + " " + str(PMFE[i]) + "\n")

with open("../Results/PMF/pmf_GE.dat", 'w') as f:
    f.write('#Rg Ree A\n')
    for i in range(len(PMFGE)):
        for j in range(len(PMFGE[0])):
            f.write(str(G2[i]) + " " + str(E2[j]) + " " + str(PMFGE[i,j]) + "\n")
