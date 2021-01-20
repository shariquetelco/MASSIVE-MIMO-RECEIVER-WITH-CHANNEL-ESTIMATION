# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:50:18 2020

@author: MOHAMMAD SHARIQUE AHMAD
ahmad@eurecom.fr
sharique.mohammad@cyberteq.com
sharique.ahmad@ieee.org

"""
import numpy as np
import numpy.linalg as nl
import numpy.random as nr
import matplotlib.pyplot as plt
#from scipy.stats import norm
import m5G
fig, ax = plt.subplots()

ITER = 1000;
K = 10; # number of users in the shell
Mv = np.arange(20,1000,60); # number of BS antennas arrays (20 to 1000 tap of 60)
Eu_dB = 10;  Eu = 10**(Eu_dB/10); #power assign to sytem 10db 
rate_MRC = np.zeros(len(Mv)) ; # this is vector calc the rate if MRC is used length is 1*array (log2 1 +SINR)
bound_MRC = np.zeros(len(Mv)); # Simulation and Therotical result

beta = m5G.Dmatrix(K); #large scale fading factor power 
sqrtD = np.diag(np.sqrt(beta)); #matrix

dftmtx = m5G.DFTmat(K); # generating DFT matrix of K*K
# to channel estimate like (TRAINING MATRIX) it work as orthogonal matrix

for it in range(ITER):
    print (it)
    for mx in range(len(Mv)):
        M =Mv[mx];
        pu = Eu; # no power scaling
        pp = Eu/np.sqrt(M); #power scaling
        Pp = K*pu;
        H = (nr.normal(size=(M,K))+1j*nr.normal(size=(M,K)))/np.sqrt(2);
        G = np.matmul(H,sqrtD);
        Phi = np.sqrt(1/K)*dftmtx; #pilot matrix
        N = (nr.normal(size=(M,K))+1j*nr.normal(size=(M,K)))/np.sqrt(2);
        RxBLK = np.sqrt(Pp)*np.matmul(G,Phi) + N;
        Ghat = np.sqrt(1/Pp)*np.matmul(RxBLK,m5G.H(Phi)); # Channel estimation
        g0hat = Ghat[:,0];
        g0 = G[:, 0];
        e0 = g0hat - g0;
        nr_MRC = pu*nl.norm(g0)**2;
        nr_bound_MRC = pu*M*beta[0];
        dr_bound_MRC =1/K +(beta[0]+1/K/pu)/beta[0];
        g0norm = g0/nl.norm(g0);
        g0hat_norm = g0hat/nl.norm(g0);
        CSIint = np.matmul(m5G.H(g0norm),e0);
        CSIint = pu*np.abs(CSIint)**2;
        nint = nl.norm(g0hat)**2/nl.norm(g0)**2;
        dr_MRC = CSIint + nint;
        dr_bound_MRC = dr_bound_MRC + pu*np.sum((beta[0]+1/K/pu)*beta[1:]/beta[0]);
        MUint= np.matmul(m5G.H(g0hat_norm),G[:,1:]);
        dr_MRC = dr_MRC + pu*nl.norm(MUint)**2;
        rate_MRC[mx] = rate_MRC[mx] + np.log2(1+nr_MRC/dr_MRC);
        bound_MRC[mx] = bound_MRC[mx] + np.log2(1 + nr_bound_MRC/dr_bound_MRC);

rate_MRC = rate_MRC/ITER;
bound_MRC = bound_MRC/ITER; 

plt.plot(Mv, rate_MRC,'g-');
plt.plot(Mv, bound_MRC,'rs');
plt.grid(1,which='both')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0.1*y1,2*y2))
plt.legend(["MRC", "MRC Bound"], loc ="upper left");
plt.suptitle('SINR MRC with CSI Estimation')
plt.ylabel('Rate--->')
plt.xlabel('Number of antennas M-->') 
fig.text(0.95, 0.05, 'Copyright Sharique',
         fontsize=21, color='white',
         ha='right', va='bottom', alpha=0.5)

#mv is an array we are accessing this array 1 by 1 we are calcualting weight for different antenna upto 1000 with step size 60
#pp pilot power
#pu is power scaling
#small scale fading coeffiicnet is generated with complex mean varinac and hannel matrix m*k
#matmul = channel for massive mimo
#phi  - we are scaling DFT MATRIX channel matrix we make it orthonormal 
#N is noise matrix - k number of pilot matrix transmiting 
#RxBLK observation (massive mimo channel matrix and adding noise. Simpley = hx+n )
#Ghat =channel estimation with phi hermition
#g0hat =we are calculating rate for 1st user so it 0th COLUMN of matrix g0Hat m antenna 1 user
#e0 = error vector of user in the shell.
#nr_mrc = numerator MRC =
#when m tends to infinity