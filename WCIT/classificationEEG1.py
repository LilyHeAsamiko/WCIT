# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 17:54:51 2021

@author: LilyHeAsamiko
"""
import qiskit
#qiskit.__version__
from qiskit import IBMQ
IBMQ.load_account()
#MY_API_TOKEN = 'a93830f80226030329fc4e2e4d78c06bdf1942ce349fcf8f5c8021cfe8bd5abb01e4205fbd7b9c34f0b26bd335de7f1bcb9a9187a2238388106d16c6672abea2'
#provider = IBMQ.enable_account(MY_API_TOKEN)

from qiskit.compiler import transpile, assemble,schedule
from qiskit.tools.jupyter import *
from qiskit.visualization import *
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import sys
import io
import requests
import urllib
#store pi amplitudes
#given the drive and target indices, and the option to either start with the drive qubit in the ground or excited state, returns a list of experiments for observing the oscillations.
from IPython import display
import time
import pandas as pd
# importing Qiskit
import qiskit
from qiskit import IBMQ, Aer
from qiskit import QuantumCircuit, execute
from qiskit.circuit import ClassicalRegister,QuantumRegister, QuantumCircuit, Gate
# import basic plot tools
from qiskit.visualization import plot_histogram
from random import *
from qiskit.visualization.bloch import Bloch
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import qiskit.pulse as pulse
import qiskit.pulse.pulse_lib as pulse_lib
from qiskit.pulse.pulse_lib import Gaussian, GaussianSquare
from qiskit.compiler import assemble
from qiskit.ignis.characterization.calibrations import rabi_schedules, RabiFitter
#from qiskit.pulse.commands import SamplePulse
from qiskit.pulse import *
from qiskit.tools.monitor import job_monitor
# function for constructing duffing models
from qiskit.providers.aer.pulse import duffing_system_model

#We will experimentally find a π-pulse for each qubit using the following procedure: 
#- A fixed pulse shape is set - in this case it will be a Gaussian pulse. 
#- A sequence of experiments is run, each consisting of a Gaussian pulse on the qubit, followed by a measurement, with each experiment in the sequence having a subsequently larger amplitude for the Gaussian pulse. 
#- The measurement data is fit, and the pulse amplitude that completely flips the qubit is found (i.e. the π-pulse amplitude).
import warnings
warnings.filterwarnings('ignore')
from qiskit.tools.jupyter import *
get_ipython().run_line_magic('matplotlib', 'inline')

provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibmq_armonk')
backend_config = backend.configuration()
assert backend_config.open_pulse, "Backend doesn't support Pulse"
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer import PulseSimulator
from scipy.optimize import curve_fit
import math

# import basic plot tools
from qiskit.visualization import plot_histogram
from scipy.signal import hilbert, chirp
from qiskit.circuit import Parameter
from scipy.stats import f_oneway

def classify(point: complex):
    """Classify the given state as |0> or |1>."""
    def distance(a, b):
        return math.sqrt((np.real(a) - np.real(b))**2 + (np.imag(a) - np.imag(b))**2)
    return int(distance(point, mean_exc) < distance(point, mean_gnd))

def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)
    
    return fitparams, y_fit

def initialize_s(qc, qubits):
    """Apply a H-gate to 'qubits' in qc"""
    # Initialise all qubits to |+>
    for q in qubits:
        qc.h(q)
    return qc

def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    # We will return the diffuser as a gate
    U_s = qc.to_gate()
    U_s.name = "U$_s$"
    return U_s

def Uw():
    Uw = np.aray([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    return Uw

def qft(n):
    """Creates an n-qubit QFT circuit"""
    circuit = QuantumCircuit(4)
    def swap_registers(circuit, n):
        for qubit in range(n//2):
            circuit.swap(qubit, n-qubit-1)
        return circuit
    def qft_rotations(circuit, n):
        """Performs qft on the first n qubits in circuit (without swaps)"""
        if n == 0:
            return circuit
        n -= 1
        circuit.h(n)
        for qubit in range(n):
            circuit.cu1(np.pi/2**(n-qubit), qubit, n)
        qft_rotations(circuit, n)
    
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

def get_job_data(job, average):
    """Retrieve data from a job that has already run.
    Args:
        job (Job): The job whose data you want.
        average (bool): If True, gets the data assuming data is an average.
                        If False, gets the data assuming it is for single shots.
    Return:
        list: List containing job result data. 
    """
    job_results = job.result(timeout=120) # timeout parameter set to 120 s
    result_data = []
    for i in range(len(job_results.results)):
        if average: # get avg data
            result_data.append(job_results.get_memory(i)[qubit]*scale_factor) 
        else: # get single data
            result_data.append(job_results.get_memory(i)[:, qubit]*scale_factor)  
    return result_data

def get_closest_multiple_of_4(num):
    """Compute the nearest multiple of 4. Needed because pulse enabled devices require 
    durations which are multiples of 4 samples.
    """
    return (int(num) - (int(num)%4))

# samples need to be multiples of 16
def get_closest_multiple_of_16(num):
    return int(num + 8 ) - (int(num + 8 ) % 16)

def qft_dagger(qc, n):
    """n-qubit QFTdagger the first n qubits in circ"""
    circ = QuantumCircuit(n)
    # Don't forget the Swaps!
    for qubit in range(n//2):
        circ.swap(qubit, n-qubit-1)
    for j in range(n):
        for k in range(j):
            circ.cu1(-np.pi/float(2**(j-k)),k,j)#circ.cx(j, j+1)
        circ.h(j)
    circ.name = 'QFT+'
    qc.append(circ,range(n))
    return qc

# center data around 0
def baseline_remove(values):
    return np.array(values) - np.mean(np.array(values))

def motion(t, rep):
    # assume t = 2， rep = 1
    # circuit estimates the phase of a unitary operator U. It estimates θ in ψ⟩=e2πiθ|ψ⟩ , where  |ψ⟩  is an eigenvector and  e2πiθ  is the corresponding eigenvalue. The circuit operates in the following steps
    # qubits 0 to 1 as counting qubits, and qubit 2 as the eigenstate of the unitary operator (T)
    
    #i. Setup:  |ψ⟩  is in one set of qubit registers. (Set it as 1)
    qpe = QuantumCircuit(t+1, t) #(3, 2)
    qpe.x(t)
    
    #ii. Superposition: Apply a n-bit Hadamard gate operation  
    # H⊗n on the counting register:
    for qubit in range(t):
        qpe.h(qubit)
    #Controlled Unitary Operations: We need to introduce the controlled X gate 
    #C−X that applies the unitary operator U on the target register only if its corresponding control bit is  
    #|1⟩ . Since U is a unitary operatory with eigenvector |ψ⟩  such that U|ψ⟩=eπiθ|ψ⟩ , this means:
    # repetitions = 1
    for counting_qubit in range(t):
        for i in range(rep):
            #qpe.cx(math.pi/4, counting_qubit, 3); This is C-U
            qpe.cx(counting_qubit, counting_qubit +1); 
        rep *= 2     
    # measure the counting register    
    qpe.barrier()
    qpe.draw()
    return qpe

def chi(w, N):
# the time needed to travel from z(0)=np.sqrt(w/2/sigma) to z(t) = D/N = np.sqrt(w*(2-w)), i.e. the point of maximum separation of the two number state [gaussian distributions] forming the macroscopic superposition state is then obtained by:
# chi**t = 2*np.log(np.sqrt(2*(2-w)/sigma**2)+np.sqrt(2*(2-w)/sigma**2)-1)/N/np.sqrt(w*(2-w))
    return 2*np.log(8*N*(2-w))/N/np.sqrt(w*(2-w))

def operatorSz():
    return np.array([[1,0],[0,np.exp(1j/np.pi/2)]])

def operatorDSz():
    return np.array([[1,0],[0,np.exp(-1j/np.pi/2)]])

def operatorA(hbar, m, x, w0, p):
    assert(np.shape(x) == np.shape(p))
    a = 1j*np.sqrt(m/2/hbar/w0)*(p/m-1j*w0*x)
    return a
    
def operatorDA(hbar, m, x, w0, p):
    assert(np.shape(x) == np.shape(p))
    a = -1j*np.sqrt(m/2/hbar/w0)*(p/m+1j*w0*x)
    return a

def operatorP(hbar, signal): #t*x
    signal = np.array(signal)
    if np.size(signal) > np.shape(signal)[0]:
        P = np.ones((np.shape(signal)))
        r,c = np.shape(signal) #cols = Dims + Labels  
        for t in range(r):
            dx = signal[:,1:c] - signal[:,0:c-1]
            P[:,1:c] = hbar/1j*dx
    else:
        l = len(signal)
        signal = np.reshape(signal,(1,l))
        dx = signal[0,1:l] - signal[0,0:l-1]
        P = np.ones((1,l))  #P  = np.ones(np.shape(signal))
        P[0,1:l] = hbar/1j*dx
    return P

def DurationDist(PSTH, fs): 
    return PSTH/fs

def ASMfr(F3, F4): 
    return F4-F3*F4+F3
 
def ASMTemp(F7, F8): 
    return F8-F7*F8+F7 

def ASMBand(Afr, At):
    return Afr+At

def SpecDensityBand(N, samples, signal):
    #single spikes
    Pnts = N
    bins = samples #samples/window
    #Ns = length(MEAy_ch(MEAy_ch > threshold));
    if np.mod(Pnts, bins) > 0:
        N_bin = int(np.ceil(Pnts/bins)) #total bins
    else:
        N_bin = int(Pnts/bins)
    
    chf_bin_d = np.zeros((N_bin, bins))
    chf_bin_th = np.zeros((N_bin, bins))
    chf_bin_a = np.zeros((N_bin, bins))
    chf_bin_m = np.zeros((N_bin, bins))
    chf_bin_s = np.zeros((N_bin, bins))
    chf_bin_b = np.zeros((N_bin, bins))
    chf_bin_g = np.zeros((N_bin, bins))
    chn_bin = np.zeros((N_bin, bins))
    MEAs_chft = np.zeros((N_bin, int(0.5*bins)))
    MEAs_chsd = np.zeros((N_bin, int(0.5*bins)))
    Ns_chn_bin = np.zeros((N_bin,1))
    Nsft_chn_bin = np.zeros((N_bin,1))
    Nssd_chn_bin = np.zeros((N_bin,1))
    MEAs_chf = np.zeros((N_bin, bins))
    #per second
    for n in range(N_bin):
    # nn = 2*n-1;
        MEAs_chn = np.zeros((bins, 1)) #           some MEAs_chn[0:len(range(bins*n,Pnts))] = np.reshape(signal[bins*n:Pnts],(1,len(signal[bins*n:Pnts])))
        if n == N_bin-1:
            if len(range(bins*n,Pnts)) < bins:
                MEAs_chn[0:len(range(bins*n,Pnts))] = np.reshape(signal[bins*n:Pnts],(len(signal[bins*n:Pnts]),1))
            else:
                MEAs_chn = signal[bins*n:Pnts]
            MEAs_chft[n,:] = np.transpose(MEAs_chn[0:int(0.5*bins)]) #range(int(0.5*bins))
            MEAs_chsd[n,:] = np.transpose(MEAs_chn[int(0.5*bins):bins])
        else:
            MEAs_chn = signal[int(bins*n):bins*(n+1)]  
            MEAs_chft[n,:] = MEAs_chn[0:int(0.5*bins)]
            MEAs_chsd[n,:] = MEAs_chn[int(0.5*bins):bins]
        #size(X)       
    #    if len(np.fft.fft(MEAs_chn))< len(MEAs_chf[n,:])：
    #        FFT_MEAs_chn = [fft(MEAs_chn') zeros(1, length(MEAs_chf(n,:))-length(fft(MEAs_chn)))];
    #        EXT_MEAs = [MEAs_chn' zeros(1, length(MEAs_chf(n,:))-length(MEAs_chn))];
    #    else
        assert(len(np.fft.fft(MEAs_chn))== len(MEAs_chf[n,:]))
        FFT_MEAs_chn = np.fft.fft(MEAs_chn)
        EXT_MEAs = MEAs_chn
        MEAs_chf[n,:] = np.reshape(FFT_MEAs_chn[:],(1,bins))
        fr = np.fft.rfftfreq(bins*2-1, d=1./samples)
        Chf_bin_d_freq = (fr > min(delta)) * (fr < max(delta))
        Chf_bin_th_freq = (fr > min(theta)) * (fr < max(theta))
        Chf_bin_a_freq = (fr > min(alpha)) * (fr < max(alpha))
        Chf_bin_m_freq = (fr > min(mu)) * (fr < max(mu))
        Chf_bin_s_freq = (fr > min(SMR)) * (fr < max(SMR))
        Chf_bin_b_freq = (fr > min(beta)) * (fr < max(beta))
        Chf_bin_g_freq = (fr > min(gamma)) * (fr < max(gamma))
    
        chf_bin_d[n,Chf_bin_d_freq] = abs(MEAs_chf[n, Chf_bin_d_freq])**2
        chf_bin_th[n,Chf_bin_th_freq] = abs(MEAs_chf[n, Chf_bin_th_freq])**2
        chf_bin_a[n,Chf_bin_a_freq] = abs(MEAs_chf[n, Chf_bin_a_freq])**2
        chf_bin_m[n,Chf_bin_m_freq] = abs(MEAs_chf[n, Chf_bin_m_freq])**2
        chf_bin_s[n,Chf_bin_s_freq] = abs(MEAs_chf[n, Chf_bin_s_freq])**2
        chf_bin_b[n,Chf_bin_b_freq] = abs(MEAs_chf[n, Chf_bin_b_freq])**2
        chf_bin_g[n, Chf_bin_g_freq] = abs(MEAs_chf[n, Chf_bin_g_freq])**2
        Ns_chn_bin[n] = sum(abs(MEAs_chf[n,:])**2 >= threshold) #len(MEAs_chn[abs(MEAs_chf[n,:])**2 >= threshold])
#        Nsft_chn_bin[n] = sum(abs(MEAs_chf[n,0:int(0.5*bins)])**2 >= threshold) #len(MEAs_chft[abs(MEAs_chf[n,:])**2 >= threshold])
#        Nssd_chn_bin[n] = sum(abs(MEAs_chf[n,int(0.5*bins):bins])**2 >= threshold)
    fig = plt.figure()
    ax1 = fig.add_subplot(711)
    ax1.pcolor(chf_bin_d)
    ax1.set_title('Delta band: stim v.s bins(per ms)')
    ax2 = fig.add_subplot(712)
    ax2.pcolor(chf_bin_th)
    ax2.set_title('Theta band: stim v.s bins(per ms)')
    ax3 = fig.add_subplot(713)
    ax3.pcolor(chf_bin_a);
    ax3.set_title('Alpha band: stim v.s bins(per ms)')
    fig1 = plt.figure(),
    ax4 = fig.add_subplot(714)
    ax4.pcolor(chf_bin_m);
    ax4.set_title('Mu band: stim v.s bins(per ms)')
    ax5 = fig.add_subplot(715)
    ax5.pcolor(chf_bin_s);
    ax5.set_title('SMR band: stim v.s bins(per ms)')
    ax6 = fig.add_subplot(716)
    ax6.pcolor(chf_bin_b);
    ax6.set_title('Beta band: stim v.s bins(per ms)')
    ax7 = fig.add_subplot(717)
    ax7.pcolor(chf_bin_g);
    ax7.set_title('Gamma band: stim v.s bins(per ms)')

    plt.figure()
    plt.pcolor(Ns_chn_bin)
    plt.title('PSTH:Delta spikes v.s bins(per ms)')
    plt.colorbar()
    
    result = [chf_bin_d,chf_bin_th,chf_bin_a,chf_bin_m, chf_bin_s,chf_bin_b,chf_bin_g,Ns_chn_bin]
    
    return result

def c_amod15(a, power):
    """Controlled multiplication by a mod 15"""
    if a not in [2,7,8,11,13]:
        raise ValueError("'a' must be 2,7,8,11 or 13")
    U = QuantumCircuit(4)        
    for iteration in range(power):
        if a in [2,13]:
            U.swap(0,1)
            U.swap(1,2)
            U.swap(2,3)
        if a in [7,8]:
            U.swap(2,3)
            U.swap(1,2)
            U.swap(0,1)
        if a == 11:
            U.swap(1,3)
            U.swap(0,2)
        if a in [7,11,13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = "%i^%i mod 15" % (a, power)
    c_U = U.control()
    return c_U

def qpe_amod15(a, n_count):
    #n_count = 3
    #n_count = 1
    qc = QuantumCircuit(4+n_count, n_count)
    for q in range(n_count):
        qc.h(q)     # Initialise counting qubits in state |+>
    qc.x(3+n_count) # And ancilla register in state |1>
    for q in range(n_count): # Do controlled-U operations
        qc.append(c_amod15(a, 2**q), 
                 [q] + [i+n_count for i in range(4)])
#    qc.append(qft_dagger(n_count), range(n_count)) # Do inverse-QFT
    qft_dagger(qc, n_count) # Do inverse-QFT
    qc.measure(range(n_count), range(n_count))
    print(qc)
    print(qc.draw())
    # Simulate Results
    backend = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend, shots=1, memory=True).result()
    # Setting memory=True below allows us to see a list of each sequential reading
#    for output in counts:
#        deciphasemal = int(output, 2)  # Convert (base 2) string to decimal
#        phase = decimal/(2**n_count) # Find corresponding eigenvalue
#        measured_phases.append(phase)
        # Add these values to the rows in our table:
#       rows.append(["%s(bin) = %i(dec)" % (output, decimal), 
#                     "%i/%i = %.2f" % (decimal, 2**n_count, phase)])
    # Alternatively, Print the rows in a table
#    headers=["Register Output", "Phase"]
#    df = pd.DataFrame(rows, columns=headers)
#    print(df)
    readings = result.get_memory()
    print("Register Reading: " + readings[0])
    phase = int(readings[0],2)/(2**n_count)
    print("Corresponding Phase: %f" % phase)
    #reproduce:
    print("Test with repeated experiments:")
    backend = Aer.get_backend('qasm_simulator')    
    results = execute(qc, backend, shots=32).result()
    # counts = results.get_counts()
    # print(plot_histogram(counts))
    # plt.show()
    output = [phase, results]
    return output

def a2jmodN(a, j, N):
    """Compute a^{2^j} (mod N) by repeated squaring"""
    for i in range(j):
        a = np.mod(a**2, N)
    return a

def sub_circ(label, N, theta, A, B):    
    qr = QuantumRegister(N, name="q")
#    cr = ClassicalRegister(1, name='c')
#    sub_circ = QuantumCircuit(qr, cr, name='sub_circ')
#    theta = 0
    f = A*np.sin(theta+B)
    sub_circ = QuantumCircuit(qr, name='sub_circ')
    n = N-1
    sub_circ.h(range(N))
    #theta = Parameter('θ')
#    sub_circ.h(range(N-1)
    for i in range(n-1):
        if label[i] == 1:
            sub_circ.x(i)
            sub_circ.u3(0, B, 1, i)
            sub_circ.cx(i, i+1)
        theta += np.pi/2 - np.arctan(f/A*np.sin(theta+B+np.pi/2))
        f = A*np.sin(theta+B)
        sub_circ.u3(0, theta, 1, i)   
#        sub_circ.u3(0, np.pi/(n), 1, i+1)   
        sub_circ.cx(i, i+1)
    if (i == n-1) & (i % 2 == 1):
        sub_circ.u3(0, theta, 1, i)             
#    sub_circ.measure_all()
#    sub_circ.draw()
#    sub_inst = sub_circ.to_instruccction()
    return [sub_circ,f,A,B,theta]
  
#        ┌───┐┌───┐       ┌───────────┐                                            ░ ┌─┐         
#   q_0: ┤ H ├┤ X ├───────┤ U3(0,π,1) ├──■─────────────────────────────────────────░─┤M├─────────
#        ├───┤└───┘       └───────────┘┌─┴─┐┌──────────────┐                       ░ └╥┘┌─┐ 
#   q_1: ┤ H ├─────────────────────────┤ X ├┤ U3(0,3π/2,1) ├──■────────────────────░──╫─┤M├──────
#        ├───┤                         └───┘└──────────────┘┌─┴─┐                  ░  ║ └╥┘┌─┐   
#   q_2: ┤ H ├──────────────────────────────────────────────┤ X ├──────────────────░──╫──╫─┤M├───
#        ├───┤                                              └───┘┌───────────┐     ░  ║  ║ └╥┘┌─┐
#   q_3: ┤ H ├───────────────────────────────────────────────────┤ U3(0,0,1) ├─────░──╫──╫──╫─┤M├
#        └───┘                                                   └───────────┘     ░  ║  ║  ║ └╥┘
#  meas: 5/═══════════════════════════════════════════════════════════════════════════╩══╩══╩══╩═
#                                                                                     0  1  2  3 
#                                                                             ░ ┌─┐            
#...  q4_0: ──────────────────────────────────────────────────────────────■───░─┤M├────────────
#...                                                     ┌─────────────┐┌─┴─┐ ░ └╥┘┌─┐         
#...  q4_1: ──────────────────────────────────────────■──┤ U3(0,π/4,1) ├┤ X ├─░──╫─┤M├─────────
#...                                 ┌─────────────┐┌─┴─┐└─────────────┘└───┘ ░  ║ └╥┘┌─┐      
#...  q4_2: ──────────────────────■──┤ U3(0,π/4,1) ├┤ X ├─────────────────────░──╫──╫─┤M├──────
#...             ┌─────────────┐┌─┴─┐└─────────────┘└───┘                     ░  ║  ║ └╥┘┌─┐   
#...  q4_3: ──■──┤ U3(0,π/4,1) ├┤ X ├─────────────────────────────────────────░──╫──╫──╫─┤M├───
#...        ┌─┴─┐└─────────────┘└───┘                                         ░  ║  ║  ║ └╥┘
#...  q4_4: ┤ X ├─────────────────────────────────────────────────────────────░──╫──╫──╫──╫────
#...        └───┘                                                             ░  ║  ║  ║  ║ 
#...  meas: 5/═══════════════════════════════════════════════════════════════════╩══╩══╩══╩══╩═
#                                                                                0  1  2  3  4                                                                                             0  1  2  3  4 

def circ1(seq, shots, t1, weight, N, theta, A, B):
    t1 += time.time()
    #order ascendently by weight
    o = sorted(range(len(weight)), key = lambda k: seq[k], reverse = False)
    newseq = []
    for i in range(len(o)):
        newseq.append(seq[o[i]])
 #   qr = QuantumRegister(N)
 #   cr = ClassicalRegister(3)
#    circ = QuantumCircuit(qr)
    C = []
    F = []
    Sinf = []
    Theta = []
    AA = []
    BB = []
    count = 0
    for j in range(0,N,4):
        sub_cOut = sub_circ(seq, j+4, theta, A, B) #cir,sin, A,B,theta 
        Sinf.append(sub_cOut[1])
        AA.append(sub_cOut[2])
        BB.append(sub_cOut[3])
        Theta.append(sub_cOut[4])
        sub_inst = sub_cOut[0].to_instruction()
        qr = QuantumRegister((j+4)%5)
        cr = ClassicalRegister((j+4)%3)
        circ0 = QuantumCircuit(qr[0:3],cr)
        circ0.append(sub_inst, qr[0:3])
        circ0.measure(qr[3],cr)
        circ = QuantumCircuit(qr,cr)
        circ.append(sub_inst, qr)
        circ.measure(qr,cr)
        qft_dagger(circ, 4)        
        circ.measure_all()
        circ.draw()
        backend = Aer.get_backend("qasm_simulator")
        results = execute(circ, backend, shots=300).result()
        counts = results.get_counts()
        C.append(counts)
        comp = 0
        for c in counts:
    #        if c <= '3':
            if int(c[0:5])<=11:
                comp += counts[c]
        if comp <= shots/2:
            count += 1 
        f = np.sqrt(comp/shots*(1-comp/shots))**2
        F.append(f) # fidelity        
         #output_distr = get_probability_distribution(counts)
        #plot_histogram(counts)
    C = np.array(C, dtype = float) #3,2^4
    F = np.array(C, dtype = float)
    t2 =time.time()
    return [C, Comp, F, t2-t1, int(count>2), Sinf, Theta, A, B]

def circ2(seq, shots, t1, weight, N, theta, A, B):
    t1 += time.time()
    #order ascendently by weight
    o = sorted(range(len(weight)), key = lambda k: seq[k], reverse = False)
    newseq = []
    for i in range(len(o)):
        newseq.append(seq[o[i]])
    qr = QuantumRegister(4)
    #cr = QuantumRegister(3)
    #circ = QuantumCircuit(qr,cr)
    cr = ClassicalRegister(1)
    C = []
    F = []
    sub_cOut = sub_circ(seq, N, theta, A, B) #cir,sin, A,B,theta 
    sinf = sub_cOut[1]
    A = sub_cOut[2]
    B = sub_cOut[3]
    theta = sub_cOut[4]
    sub_inst = sub_cOut[0].to_instruction()
    circ0 = QuantumCircuit(qr,cr)
    circ0.append(sub_inst, qr[0:4])
    circ0.measure(qr[-1],cr)
    circ = QuantumCircuit(qr,cr)
    circ.append(sub_inst, qr[0:4])
    circ.measure(qr[-1],cr)
    qft_dagger(circ, 4)        
    circ.measure_all()
    circ.draw()
    backend = Aer.get_backend("qasm_simulator")
    results = execute(circ, backend, shots=300).result()
    counts = results.get_counts()
#    C.append(C/shots)
    comp = 0
    count = 0
    for c in counts:
#        if c <= '3':
        if int(c[0:5])<=11:
            comp += counts[c]
    if comp <= shots/2:
        count = 1 
    f = np.sqrt(comp/shots*(1-comp/shots))**2
    F.append(f) # fidelity        
    #output_distr = get_probability_distribution(counts)
    #plot_histogram(counts)
    C = np.array(C, dtype = float) #3,2^4
    F = np.array(C, dtype = float)
    t2 =time.time()
    return [counts, comp, F, t2-t1, count, sinf, theta, A, B]

#        ┌───────────┐    ░ ┌─┐         
#  q0_0: ┤0          ├────░─┤M├─────────
#        │           │    ░ └╥┘┌─┐      
#  q0_1: ┤1          ├────░──╫─┤M├──────
#        │  sub_circ │    ░  ║ └╥┘┌─┐   
#  q0_2: ┤2          ├────░──╫──╫─┤M├───
#        │           │┌─┐ ░  ║  ║ └╥┘┌─┐
#  q0_3: ┤3          ├┤M├─░──╫──╫──╫─┤M├
#        └───────────┘└╥┘ ░  ║  ║  ║ └╥┘
#  c0: 1/══════════════╩═════╬══╬══╬══╬═
#                      0     ║  ║  ║  ║ 
#                            ║  ║  ║  ║ 
#meas: 4/════════════════════╩══╩══╩══╩═
#                            0  1  2  3 

#        ┌───────────┐   ┌───────┐ ░ ┌─┐         
# q16_0: ┤0          ├───┤0      ├─░─┤M├─────────
#        │           │   │       │ ░ └╥┘┌─┐      
# q16_1: ┤1          ├───┤1      ├─░──╫─┤M├──────
#        │  sub_circ │   │  QFT+ │ ░  ║ └╥┘┌─┐   
# q16_2: ┤2          ├───┤2      ├─░──╫──╫─┤M├───
#        │           │┌─┐│       │ ░  ║  ║ └╥┘┌─┐
# q16_3: ┤3          ├┤M├┤3      ├─░──╫──╫──╫─┤M├
#        └───────────┘└╥┘└───────┘ ░  ║  ║  ║ └╥┘
#  c2: 1/══════════════╩══════════════╬══╬══╬══╬═
#                                     ║  ║  ║  ║ 
#meas: 4/═════════════════════════════╩══╩══╩══╩═
#                                     0  1  2  3                                                                                                                         
hbar = 2.1091436/2 * 10**(-34)#h/pi = 2.1091436 × 10-34 m2 kg / s
#readin data
url1 = 'http://homepages.cae.wisc.edu/~ece539/data/eeg/nic23a1.txt'
nic23a1 = urllib.request.urlopen(url1)
#s1=requests.get(url1).content
#nic23a1=pd.read_csv(io.StringIO(s1.decode('utf-8')))
#nic23a1 = pd.read_csv(url1, delimiter='\n')
url2 = 'http://homepages.cae.wisc.edu/~ece539/data/eeg/nic23a3.txt'
nic23a3 = urllib.request.urlopen(url2)
#another obervations
url11 = 'http://homepages.cae.wisc.edu/~ece539/data/eeg/nic8a1.txt'
nic8a1 = urllib.request.urlopen(url11)
url21 = 'http://homepages.cae.wisc.edu/~ece539/data/eeg/nic8a3.txt'
nic8a3 = urllib.request.urlopen(url21)
Dims = 29
Labels = 8
# Do not consider the effect of time first(Spacially only)
Width = int(Dims/Labels)+1
tx1 = []
tx2 = []
#print(np.shape(nic23a1.readlines()))
for line1 in nic23a1.readlines():
    tx1.append(line1.split())
for line2 in nic23a3.readlines():
    tx2.append(line2.split())
tx1 = np.array(tx1, dtype = float) #raw: t, col: x(29)+features(8)
tx2 = np.array(tx2, dtype = float) #same data as tx1 but with different labels
rows,cols = np.shape(tx1) #cols = Dims + Labels
print(rows)
print(cols)
dataset = []
label1 = []
label2 = []
for i in range(rows):
    dataset.append(tx1[i][range(Dims)])
    label1.append(tx1[i][range(Dims,Dims+Labels)])
    label2.append(tx2[i][range(Dims,Dims+Labels)])
label1 = np.array(label1,dtype = float)
label2 = np.array(label2,dtype = float)
dataset= np.array(dataset,dtype = float)

#preprocess: Hilbert transform
samples = 200
Samples = rows
fs = 2000 #kHz
T = samples/fs #s, samples = int(fs*duration)
duration = Samples/fs
t = np.arange(Samples) / fs
dt = np.arange(samples)/ fs
temp = rows

Signal = []
for c in range(Dims):
    signal = dataset[:,c]
    signalk = [] 
    analytic_signal = hilbert(dataset[:,c])
    for steps in range(int(fs/samples)): 
        if np.mean(signal) !=0:
            for steps in range(samples):              
                amplitude_envelope = np.abs(analytic_signal)
                instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                instantaneous_frequency = (np.diff(instantaneous_phase) /
                                           (2.0*np.pi) * fs)
                if abs(sum(amplitude_envelope>np.mean(analytic_signal)) - sum(amplitude_envelope<=np.mean(analytic_signal))) <= 1:
                    break
                else:
                    analytic_signal -= np.mean(analytic_signal)
            print(abs(sum(abs(amplitude_envelope - analytic_signal) <= np.mean(abs(analytic_signal - np.mean(signal))))) - sum(analytic_signal == 0))
            if abs(sum(abs(amplitude_envelope - analytic_signal) <= np.mean(abs(analytic_signal - np.mean(signal)))) - sum(analytic_signal == 0)) <= temp:
                temp = abs(sum(abs(amplitude_envelope - analytic_signal) <= np.mean(abs(analytic_signal - np.mean(signal)))) - sum(analytic_signal == 0))
            if temp <= 1:
                signalk.append(analytic_signal)
                UppL = 1.5*max(instantaneous_frequency)
                LowL = min(instantaneous_frequency)-0.1*UppL
                fig = plt.figure()
                ax0 = fig.add_subplot(211)
                ax0.plot(t, analytic_signal, label='signal')
                ax0.plot(t, amplitude_envelope, label='envelope')
                ax0.set_xlabel("time in seconds")
                ax0.legend()
                ax1 = fig.add_subplot(212)
                ax1.plot(t[1:], instantaneous_frequency)
                ax1.set_xlabel("time in seconds")
                ax1.set_ylim(LowL, UppL)
                ax1.set_title('instantaneous_frequency')            
                signal -= np.mean(analytic_signal)
        elif sum(abs(signal - 0) > 1) != 0:
            print(sum(abs(signal - 0) > 1))
            signalk.append(signal-np.mean(signal))
            analytic_signal = hilbert(signal-np.mean(signal))
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = (np.diff(instantaneous_phase) /
                                       (2.0*np.pi) * fs)
            UppL = 1.5*max(instantaneous_frequency)
            LowL = min(instantaneous_frequency)-0.1*UppL
            fig = plt.figure()
            ax0 = fig.add_subplot(211)
            ax0.plot(t, analytic_signal, label='signal')
            ax0.plot(t, amplitude_envelope, label='envelope')
            ax0.set_xlabel("time +in seconds")
            ax0.legend()
            ax1 = fig.add_subplot(212)
            ax1.plot(t[1:], instantaneous_frequency)
            ax1.set_xlabel("time in seconds")
            ax1.set_ylim(LowL, UppL)
            ax1.set_title('instantaneous_frequency')
            signal -= np.mean(signal)
    Signal.append(signalk)
    
tsignal = dataset[:,0] 
analytic_signal = hilbert(tsignal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) /
                           (2.0*np.pi) * fs)

UppL = 1.5*max(np.max(instantaneous_frequency,0))
LowL = min(np.min(instantaneous_frequency,0))-0.1*UppL
fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(t, tsignal, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
#ax0.legend()
ax1 = fig.add_subplot(212)
ax1.plot(t, instantaneous_frequency)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(LowL, UppL)
ax1.set_title('instantaneous_frequency before EMD')

#different bands
N = rows
threshold = 400
delta = np.linspace(0.5, 3, samples)
theta = np.linspace(4, 7, samples)
alpha = np.linspace(8, 12, samples)
mu = np.linspace(7.5, 12.5, samples)
SMR = np.linspace(12.5, 15.5, samples)
beta = np.linspace(16, 31, samples)
gamma = np.linspace(32, 100, samples)

L = len(dt)-1; # Signal length
Sigma = np.std(signal[range(L)])
X = 1/(4*np.sqrt(2*np.pi*Sigma))*(np.exp(-dt**2/(2*Sigma)));

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dt,X)
ax.set_title('half Gaussian Pulse in Time Domain dt')

#single spikes
Pnts = N
bins = samples #samples/window
#Ns = length(MEAy_ch(MEAy_ch > threshold));
if np.mod(Pnts, bins) > 0:
    N_bin = int(np.ceil(Pnts/bins)) #total bins
else:
    N_bin = int(Pnts/bins)
Signal = dataset
chf_bin_d = np.zeros((N_bin, bins))
chf_bin_th = np.zeros((N_bin, bins))
chf_bin_a = np.zeros((N_bin, bins))
chf_bin_m = np.zeros((N_bin, bins))
chf_bin_s = np.zeros((N_bin, bins))
chf_bin_b = np.zeros((N_bin, bins))
chf_bin_g = np.zeros((N_bin, bins))
chn_bin = np.zeros((N_bin, bins))
MEAs_chft = np.zeros((N_bin, int(0.5*bins)))
MEAs_chsd = np.zeros((N_bin, int(0.5*bins)))
Ns_chn_bin = np.zeros((N_bin,1))
Nsft_chn_bin = np.zeros((N_bin,1))
Nssd_chn_bin = np.zeros((N_bin,1))
MEAs_chf = np.zeros((N_bin, bins))
#fn = range(bins)
#f = 1.0*np.array(fn,dtype = float)/float(fs-1)
fnr = np.fft.rfftfreq(rows*2, d=1./samples)
#Chf_bin_d = (f > (min(delta)*duration/fs)) * (f < (max(delta)*duration/fs))
Chf_bin_d = (fnr > min(delta)) * (fnr < max(delta))
Chf_bin_th = (fnr > min(theta)) * (fnr < max(theta))
Chf_bin_a = (fnr > min(alpha)) * (fnr < max(alpha))
Chf_bin_m = (fnr > min(mu)) * (fnr < max(mu))
Chf_bin_s = (fnr > min(SMR)) * (fnr < max(SMR))
Chf_bin_b = (fnr > min(beta)) * (fnr < max(beta))
Chf_bin_g = (fnr > min(gamma)) * (fnr < max(gamma))

#per second
for n in range(N_bin):
# nn = 2*n-1;
    MEAs_chn = np.zeros((bins, 1)) #           some MEAs_chn[0:len(range(bins*n,Pnts))] = np.reshape(signal[bins*n:Pnts],(1,len(signal[bins*n:Pnts])))
#
    signal = Signal[n,:]
    if n == N_bin-1:
        if len(range(bins*n,Pnts)) < bins:
            MEAs_chn[0:len(range(bins*n,Pnts))] = np.reshape(signal[bins*n:Pnts],(len(signal[bins*n:Pnts]),1))
        else:
            MEAs_chn = signal[bins*n:Pnts]
        MEAs_chft[n,:] = np.transpose(MEAs_chn[0:int(0.5*bins)]) #range(int(0.5*bins))
        MEAs_chsd[n,:] = np.transpose(MEAs_chn[int(0.5*bins):bins])
    else:
        MEAs_chn = signal[int(bins*n):bins*(n+1)]  
        MEAs_chft[n,0:int(0.5*bins)] = MEAs_chn[0:int(0.5*bins)]
        MEAs_chsd[n, int(0.5*bins):bins] = MEAs_chn[int(0.5*bins):bins]
    #size(X)       
#    if len(np.fft.fft(MEAs_chn))< len(MEAs_chf[n,:])：
#        FFT_MEAs_chn = [fft(MEAs_chn') zeros(1, length(MEAs_chf(n,:))-length(fft(MEAs_chn)))];
#        EXT_MEAs = [MEAs_chn' zeros(1, length(MEAs_chf(n,:))-length(MEAs_chn))];
#    else
    assert(len(np.fft.fft(MEAs_chn))== len(MEAs_chf[n,:]))
    FFT_MEAs_chn = np.fft.fft(MEAs_chn)
    EXT_MEAs = MEAs_chn
    MEAs_chf[n,:] = np.reshape(FFT_MEAs_chn[:],(1,bins))
    fr = np.fft.rfftfreq(bins*2-1, d=1./samples)
    Chf_bin_d_freq = (fr > min(delta)) * (fr < max(delta))
    Chf_bin_th_freq = (fr > min(theta)) * (fr < max(theta))
    Chf_bin_a_freq = (fr > min(alpha)) * (fr < max(alpha))
    Chf_bin_m_freq = (fr > min(mu)) * (fr < max(mu))
    Chf_bin_s_freq = (fr > min(SMR)) * (fr < max(SMR))
    Chf_bin_b_freq = (fr > min(beta)) * (fr < max(beta))
    Chf_bin_g_freq = (fr > min(gamma)) * (fr < max(gamma))

#        MEAs_chft[n,:] = np.transpose(MEAs_chn[0:int(0.5*bins)]) #range(int(0.5*bins))
# %test
# MEAs_testn = MEA_test(bin*(n-1)+1:Pnts);
# MEAs_testft = MEA_test(0.5*bin*(nn-1)+1:0.5*bin*nn);
# MEAs_testsd = MEA_test(0.5*bin*nn+1:Pnts);

#size(fft(MEAs_chn'))
    chf_bin_d[n,Chf_bin_d_freq] = abs(MEAs_chf[n, Chf_bin_d_freq])**2
    chf_bin_th[n,Chf_bin_th_freq] = abs(MEAs_chf[n, Chf_bin_th_freq])**2
    chf_bin_a[n,Chf_bin_a_freq] = abs(MEAs_chf[n, Chf_bin_a_freq])**2
    chf_bin_m[n,Chf_bin_m_freq] = abs(MEAs_chf[n, Chf_bin_m_freq])**2
    chf_bin_s[n,Chf_bin_s_freq] = abs(MEAs_chf[n, Chf_bin_s_freq])**2
    chf_bin_b[n,Chf_bin_b_freq] = abs(MEAs_chf[n, Chf_bin_b_freq])**2
    chf_bin_g[n, Chf_bin_g_freq] = abs(MEAs_chf[n, Chf_bin_g_freq])**2
    Ns_chn_bin[n] = sum(abs(MEAs_chf[n,:])**2 >= threshold) #len(MEAs_chn[abs(MEAs_chf[n,:])**2 >= threshold])
    Nsft_chn_bin[n] = sum(abs(MEAs_chf[n,0:int(0.5*bins)])**2 >= threshold) #len(MEAs_chft[abs(MEAs_chf[n,:])**2 >= threshold])
    Nssd_chn_bin[n] = sum(abs(MEAs_chf[n,int(0.5*bins):bins])**2 >= threshold)
#    chn_bin[n,abs(MEAs_chf[n,:])**2 >= threshold] = EXT_MEAs[abs(MEAs_chf[n,:])**2 >= threshold]

#%!!
#% chf_bin_d(n,:) = abs(MEAs_chf(n,:)) > delta(1) & abs(MEAs_chf(n,:)) < delta(2);
#% chf_bin_th(n,:) = abs(MEAs_chf(n,:)) > theta(1) & abs(MEAs_chf(n,:)) < theta(2);
#% chf_bin_a(n,:) = abs(MEAs_chf(n,:)) > alpha(1) & abs(MEAs_chf(n,:)) < alpha(2);
#% chf_bin_m(n,:) = abs(MEAs_chf(n,:)) > mu(1) & abs(MEAs_chf(n,:)) < mu(2);
#% chf_bin_s(n,:) = abs(MEAs_chf(n,:)) > SMR(1) & abs(MEAs_chf(n,:)) < SMR(2);
#% chf_bin_b(n,:) = abs(MEAs_chf(n,:)) > beta(1) & abs(MEAs_chf(n,:)) < beta(2);
#% chf_bin_g(n,:) = abs(MEAs_chf(n,:)) > gamma(1) & abs(MEAs_chf(n,:)) < gamma(2);
#    chf_bin_d[n,Chf_bin_d] = abs(MEAs_chf[n, Chf_bin_d])**2
    

Ns = sum(Ns_chn_bin);
fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(t, signal, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1 = fig.add_subplot(212)
ax1.plot(t[1:], instantaneous_frequency)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(LowL, UppL)
ax1.set_title('instantaneous_frequency')

fig = plt.figure()
ax1 = fig.add_subplot(711)
ax1.pcolor(chf_bin_d)
ax1.set_title('Delta band: stim v.s bins(per ms)')
ax2 = fig.add_subplot(712)
ax2.pcolor(chf_bin_th)
ax2.set_title('Theta band: stim v.s bins(per ms)')
ax3 = fig.add_subplot(713)
ax3.pcolor(chf_bin_a);
ax3.set_title('Alpha band: stim v.s bins(per ms)')
fig1 = plt.figure(),
ax4 = fig.add_subplot(714)
ax4.pcolor(chf_bin_m);
ax4.set_title('Mu band: stim v.s bins(per ms)')
ax5 = fig.add_subplot(715)
ax5.pcolor(chf_bin_s);
ax5.set_title('SMR band: stim v.s bins(per ms)')
ax6 = fig.add_subplot(716)
ax6.pcolor(chf_bin_b);
ax6.set_title('Beta band: stim v.s bins(per ms)')
ax7 = fig.add_subplot(717)
ax7.pcolor(chf_bin_g);
ax7.set_title('Gamma band: stim v.s bins(per ms)')

fig2 = plt.figure(),
ax0 = fig2.add_subplot(311)
ax0.pcolor(Ns_chn_bin)
ax0.colorbar()
ax0.set_title('PSTH: spikes v.s bins(per ms)')
ax1 = fig2.add_subplot(312)
ax1.pcolor(Nsft_chn_bin);
ax1.set_title('PSTH: spikes number(first half) v.s time(per ms)')
ax1.colorbar()
ax2 = fig2.add_subplot(323)
ax2.pcolor(Nssd_chn_bin);
ax2.set_title('PSTH: spikes number(second half) v.s time(per ms)')
ax2.colorbar()

dur = DurationDist(Ns_chn_bin, fs)

AlphaF3 = chf_bin_a
BetaF3 = chf_bin_b  

for c in [2,7,11]:
    tempsignal = dataset[:,c]
    N = rows, 
    OutPut = SpecDensityBand(N, samples, tempsignal)
    if c == 2:
        AlphaF4 = OutPut[2]
        BetaF4 = OutPut[5]
    elif c == 7:
        AlphaF7 = OutPut[2]
        BetaF7 = OutPut[5]
    elif c == 11:
        AlphaF8 = OutPut[2]
        BetaF8 = OutPut[5]

ASMFRa = ASMfr(AlphaF3, AlphaF4) #0,2
ASMFRb = ASMfr(BetaF3, BetaF4) 
 
ASMTa = ASMTemp(AlphaT7, AlphaT8) #7,11
ASMTb = ASMTemp(BetaT7, BetaT8)

ASMa = ASMBand(ASMFRa, ASMTa)
ASMb = ASMBand(ASMFRb, ASMTb)

#test for preprocessed signal on {F3, F4}, {T7, T8} 
#The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean. The test is applied to samples from two or more groups, possibly with differing sizes.

inda = [0,2]
indb = [7,11]
Fab, pab = f_oneway(Signal[:,inda],Signal[:,indb])
#Fab: array([5363.80253618,   20.5186549 ])
#pab: array([0.00000000e+00, 6.29529926e-06])

#test for preprocessed signal on Frontal and Temporal
#Calculate a Spearman correlation coefficient with associated p-value.
#The Spearman rank-order correlation coefficient is a nonparametric measure of the monotonicity of the relationship between two datasets. Unlike the Pearson correlation, the Spearman correlation does not assume that both datasets are normally distributed
from scipy import stats
rhoFRT, pFRT = stats.spearmanr(np.array([ASMFRa[:], ASMFRb[:]]).flatten(),np.array([ASMTa[:], ASMTb[:]]).flatten())
rhoAB, pAB = stats.spearmanr(np.array([ASMFRa[:], ASMTa[:]]).flatten(),np.array([ASMFRb[:], ASMTb[:]]).flatten())
FRa = np.array(ASMFRa[:]).flatten()
FRb = np.array(ASMFRb[:]).flatten()
asma = [FRa,Ta]
Ta = np.array(ASMTa[:]).flatten()
Tb = np.array(ASMTb[:]).flatten()
asmb = [FRb,Tb]
rhoFR, pT = stats.spearmanr(np.transpose(asma),np.transpose(asmb))
#rhoFRT:
#array([[ 1.        ,  0.748671  , -0.06480895, -0.06093976],
#       [ 0.748671  ,  1.        , -0.06862124, -0.06452445],
#       [-0.06480895, -0.06862124,  1.        ,  0.75546679],
#       [-0.06093976, -0.06452445,  0.75546679,  1.        ]])
#pFRT:
#array([[0.00000000e+000, 2.29108486e-180, 4.04591829e-002,
#        5.40463756e-002],
#       [2.29108486e-180, 0.00000000e+000, 3.00191107e-002,
#        4.13487034e-002],
#       [4.04591829e-002, 3.00191107e-002, 0.00000000e+000,
#        1.80396472e-185],
#       [5.40463756e-002, 4.13487034e-002, 1.80396472e-185,
#        0.00000000e+000]])

#test for preprocessed signal on band Alpha and band Beta
resFRa = stats.linregress(ASMFRa.flatten(), ASMa.flatten())
#LinregressResult(slope=1.2433734013493718, intercept=-1851.38975656414, rvalue=0.9671919058969185, pvalue=0.0, stderr=0.010338023472535306)
resFRb = stats.linregress(ASMFRb.flatten(), ASMb.flatten())
#LinregressResult(slope=1.2228778144193009, intercept=-10985.633548078924, rvalue=0.7543490748617636, pvalue=1.279790995902578e-184, stderr=0.033687023651497845)
resTa = stats.linregress(ASMTa.flatten(), ASMa.flatten())
#LinregressResult(slope=2.4670629048159065, intercept=-4371.0211669516375, rvalue=0.7816340893321466, pvalue=8.330297401787367e-207, stderr=0.06231794707878986)
resTb = stats.linregress(ASMTb.flatten(), ASMb.flatten())
#LinregressResult(slope=1.1885247494496447, intercept=-16974.08989560993, rvalue=0.7971623151618871, pvalue=6.407795013910786e-221, stderr=0.02849470554146385)

plt.plot(ASMFRa.flatten(), ASMa.flatten(), 'o', label='original data')
plt.plot(ASMFRa.flatten(), resFRa.intercept + resFRa.slope*ASMFRa.flatten(), 'r', label='fitted line')
plt.legend()
plt.title('Alpha Frontal')
plt.show()

plt.plot(ASMTa.flatten(), ASMa.flatten(), 'o', label='original data')
plt.plot(ASMTa.flatten(), resTa.intercept + resTa.slope*ASMTa.flatten(), 'r', label='fitted line')
plt.legend()
plt.title('Alpha Temporal')
plt.show()

plt.plot(ASMFRb.flatten(), ASMb.flatten(), 'o', label='original data')
plt.plot(ASMFRb.flatten(), resFRb.intercept + resFRb.slope*ASMFRb.flatten(), 'r', label='fitted line')
plt.legend()
plt.title('Beta Frontal')
plt.show()

plt.plot(ASMTb.flatten(), ASMb.flatten(), 'o', label='original data')
plt.plot(ASMTb.flatten(), resTb.intercept + resTb.slope*ASMTb.flatten(), 'r', label='fitted line')
plt.legend()
plt.title('Beta Temporal')
plt.show()

#Calculate 95% confidence interval on slope and intercept:
from scipy.stats import t
tinv = lambda p, df: abs(t.ppf(p/2, df))
ts = tinv(0.05, len(ASMFRa.flatten())-2)
print(f"slope (95%): {resFRa.slope:.6f} +/- {ts*resFRa.stderr:.6f}")
#slope (95%): 1.243373 +/- 0.020287
print(f"intercept (95%): {resFRa.intercept:.6f}"
      f" +/- {ts*resFRa.stderr:.6f}")
#intercept (95%): -1851.389757 +/- 0.010143

tinv = lambda p, df: abs(t.ppf(p/2, df))
ts = tinv(0.05, len(ASMTa.flatten())-2)
print(f"slope (95%): {resTa.slope:.6f} +/- {ts*resTa.stderr:.6f}")
#slope (95%): 2.467063 +/- 0.122289
print(f"intercept (95%): {resTa.intercept:.6f}"
      f" +/- {ts*resTsa.tderr:.6f}")
#intercept (95%): -4371.021167 +/- 0.061145

tinv = lambda p, df: abs(t.ppf(p/2, df))
ts = tinv(0.05, len(ASMFRb.flatten())-2)
print(f"slope (95%): {resFRb.slope:.6f} +/- {ts*resFRb.stderr:.6f}")
#slope (95%): slope (95%): 1.222878 +/- 0.066106
print(f"intercept (95%): {resFRb.intercept:.6f}"
      f" +/- {ts*resFRb.stderr:.6f}")
#intercept (95%): -10985.633548 +/- 0.033053

tinv = lambda p, df: abs(t.ppf(p/2, df))
ts = tinv(0.05, len(ASMTb.flatten())-2)
print(f"slope (95%): {resTb.slope:.6f} +/- {ts*resTb.stderr:.6f}")
#slope (95%): 1.188525 +/- 0.055916
print(f"intercept (95%): {resTb.intercept:.6f}"
      f" +/- {ts*resTb.stderr:.6f}")
#intercept (95%): -16654.500426 +/- 0.055916


#shor's example
# N = 35 
# m = 4
# a = 3 #
# r = 2 #(r,f):  （1，2），（2，8），（3，8）
# f = 8
 
# # Calculate the plotting data
# xvals = np.arange(35)
# yvals = [np.mod(a**x, N) for x in xvals]

# # Use matplotlib to display it nicely
# fig, ax = plt.subplots()
# ax.plot(xvals, yvals, linewidth=1, linestyle='dotted', marker='x')
# ax.set(xlabel='$x$', ylabel='$%i^x$ mod $%i$' % (a, N),
#        title="Periodic Function in Shor's Algorithm")
# try: # plot r on the graph
#     r = yvals[1:].index(1) +1 
#     plt.annotate(text='', xy=(0,1), xytext=(r,1), arrowprops=dict(arrowstyle='<->'))
#     plt.annotate(text='$r=%i$' % r, xy=(r/3,1.5))
# except:
#     print('Could not find period, check a < N and have no common factors.')

# ax.set(xlabel='Number of applications of U', ylabel='End state of register',
#        title="Effect of Successive Applications of U")
# fig

N = 15 
m = 4
a = 2 #b = 4
#r = 1 #(r,f):  （1，1），（2，1）, (3,7), (4,1), (5,0)
f = 1

  
# Calculate the plotting data
xvals = np.arange(15)
yvals = [np.mod(a**x, N) for x in xvals]

# Use matplotlib to display it nicely
fig, ax = plt.subplots()
ax.plot(xvals, yvals, linewidth=1, linestyle='dotted', marker='x')
ax.set(xlabel='$x$', ylabel='$%i^x$ mod $%i$' % (a, N),
       title="Periodic Function in Shor's Algorithm")
try: # plot r on the graph
    r = yvals[1:].index(1) +1 
    plt.annotate(text='', xy=(0,1), xytext=(r,1), arrowprops=dict(arrowstyle='<->'))
    plt.annotate(text='$r=%i$' % r, xy=(r/3,1.5))
except:
    print('Could not find period, check a < N and have no common factors.')

ax.set(xlabel='Number of applications of U', ylabel='End state of register',
       title="Effect of Successive Applications of U")
fig

# Specify variables(2:2)
n_count = 1 # number of counting qubits
a = 2

# Create QuantumCircuit with n_count counting qubits
# plus 4 qubits for U to act on
qc = QuantumCircuit(n_count + 4, n_count)

# Initialise counting qubits
# in state |+>
for q in range(n_count):
    qc.h(q)
    
# And ancilla register in state |1>
qc.x(3+n_count)

# Do controlled-U operations
for q in range(n_count):
    qc.append(c_amod15(a, 2**q), 
             [q] + [i+n_count for i in range(4)])

# Do inverse-QFT
# qc.append(qft_dagger(n_count), range(n_count))
qft_dagger(qc,n_count)

# Measure circuit
qc.measure(range(n_count), range(n_count))
qc.draw('text')

#     ┌───┐               ┌──────┐┌─┐
#q_0: ┤ H ├───────■───────┤ QFT+ ├┤M├
#     └───┘┌──────┴──────┐└──────┘└╥┘
#q_1: ─────┤0            ├─────────╫─
#          │             │         ║ 
#q_2: ─────┤1            ├─────────╫─
#          │  2^1 mod 15 │         ║ 
#q_3: ─────┤2            ├─────────╫─
#     ┌───┐│             │         ║ 
#q_4: ┤ X ├┤3            ├─────────╫─
#     └───┘└─────────────┘         ║ 
#c: 1/═════════════════════════════╩═
 

backend = Aer.get_backend('qasm_simulator')
results = execute(qc, backend, shots=2048).result()
counts = results.get_counts()
plot_histogram(counts)

rows, measured_phases = [], []
for output in counts:
    decimal = int(output, 2)  # Convert (base 2) string to decimal
    phase = decimal/(2**n_count) # Find corresponding eigenvalue
    measured_phases.append(phase)
    # Add these values to the rows in our table:
    rows.append(["%s(bin) = %i(dec)" % (output, decimal), 
                 "%i/%i = %.2f" % (decimal, 2**n_count, phase)])
# Print the rows in a table
headers=["Register Output", "Phase"]
df = pd.DataFrame(rows, columns=headers)
print(df)

      
#   Register Output       Phase
#0  0(bin) = 0(dec)  0/2 = 0.00
#1  1(bin) = 1(dec)  1/2 = 0.50                           0 

#test reproduable
np.random.seed(1) # This is to make sure we get reproduceable results
a = randint(2, 15)
print(a)

from fractions import Fraction
from math import gcd # greatest common divisor
if gcd(a, 15) ==1:    
    np.random.seed(1) # This is to make sure we get reproduceable results
    phase = qpe_amod15(a, n_count) # Phase = s/r
    Fraction(phase).limit_denominator(15) # Denominator should (hopefully!) tell us r
    
    frac = Fraction(phase).limit_denominator(15)
    s, r = frac.numerator, frac.denominator
    print(r)
    guesses = [gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)]
    print(guesses)

#Register Reading: 1
#Corresponding Phase: 0.500000

#set parameter into 3:2
a = 2 #a = 7, r = 4
n_count = 3
factor_found = False
attempt = 0
while not factor_found:
    attempt += 1
    print("\nAttempt %i:" % attempt)
    output = qpe_amod15(a, n_count) # Phase = s/r    
    phase = output[0]
    Results = output[1]
    counts = Results.get_counts()
    plot_histogram(counts)
    plt.show()
    frac = Fraction(phase).limit_denominator(N) # Denominator should (hopefully!) tell us r
    s, r = frac.numerator, frac.denominator
#    print(s)
#    print(r)
#    print("Result: s, r = {}".format(s,r))
    print("Result: s= %i" % s)
    print("r= %i " % r )
    if phase != 0:
        # Guesses for factors are gcd(x^{r/2} ±1 , 15)
        guesses = [gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)]
        print("Guessed Factors: %i and %i" % (guesses[0], guesses[1]))
        for guess in guesses:
            if guess != 1 and (15 % guess) == 0: # Check to see if guess is a factor
                print("*** Non-trivial factor found: %i ***" % guess)
                factor_found = True

#     ┌───┐                                             ┌───────┐┌─┐      
#q_0: ┤ H ├───────■─────────────────────────────────────┤0      ├┤M├──────
#     ├───┤       │                                     │       │└╥┘┌─┐   
#q_1: ┤ H ├───────┼──────────────■──────────────────────┤1 QFT+ ├─╫─┤M├───
#     ├───┤       │              │                      │       │ ║ └╥┘┌─┐
#q_2: ┤ H ├───────┼──────────────┼──────────────■───────┤2      ├─╫──╫─┤M├
#     └───┘┌──────┴──────┐┌──────┴──────┐┌──────┴──────┐└───────┘ ║  ║ └╥┘
#q_3: ─────┤0            ├┤0            ├┤0            ├──────────╫──╫──╫─
#          │             ││             ││             │          ║  ║  ║ 
#q_4: ─────┤1            ├┤1            ├┤1            ├──────────╫──╫──╫─
#          │  2^1 mod 15 ││  2^2 mod 15 ││  2^4 mod 15 │          ║  ║  ║ 
#q_5: ─────┤2            ├┤2            ├┤2            ├──────────╫──╫──╫─
#     ┌───┐│             ││             ││             │          ║  ║  ║ 
#q_6: ┤ X ├┤3            ├┤3            ├┤3            ├──────────╫──╫──╫─
#     └───┘└─────────────┘└─────────────┘└─────────────┘          ║  ║  ║ 
#c: 3/════════════════════════════════════════════════════════════╩══╩══╩═

#Attempt 1:
#Register Reading: 011
#Corresponding Phase: 0.250000
#Test with repeated experiments:
#Result: s= 1
#r= 4 
#Guessed Factors: 3 and 5  (b-1) , (b+1)
#*** Non-trivial factor found: 3 ***
#*** Non-trivial factor found: 5 ***
          
a2jmodN(a, 1, 15)

N = 1 #{0,1}    
m= 1
qubit = 0 # for caliration 

um = 1.0e-6 # MicroMeter
wavelength = 1*um

w = 1 #consider w = 1 criticle coupling
gamma = 2*np.arcsin(np.sqrt(w*(2-w)))
D = N*np.sqrt(w*(2-w))
#Sz = operatorSz()
w1 = np.ones((1,bins))
w1[0,1:] = np.pi/dt[1:]


qr = QuantumRegister(12)
#cr = QuantumRegister(3)
#circ = QuantumCircuit(qr,cr)
circ = QuantumCircuit(qr)
C = []
tol = 0.01
for i in [0,4,8]:
    sub_c = sub_circ([1,0,1,1], 5)
    sub_inst = sub_c.to_instruction()
    circ.append(sub_inst, qr[i:i+4])
    circ.measure_all()
    backend = Aer.get_backend("qasm_simulator")
    shots = 100
    results = execute(sub_c, backend, shots=shots).result()
    counts = results.get_counts(sub_c)
    QuantumRegister(1)
    C.append(C/shots)
    #output_distr = get_probability_distribution(counts)
    plot_histogram(counts)
circ.draw()

#           ┌───────────┐ ░ ┌─┐                        ░ ┌─┐                     »
#    q61_0: ┤0          ├─░─┤M├────────────────────────░─┤M├─────────────────────»
#           │           │ ░ └╥┘┌─┐                     ░ └╥┘┌─┐                  »
#    q61_1: ┤1          ├─░──╫─┤M├─────────────────────░──╫─┤M├──────────────────»
#           │  sub_circ │ ░  ║ └╥┘┌─┐                  ░  ║ └╥┘┌─┐               »
#    q61_2: ┤2          ├─░──╫──╫─┤M├──────────────────░──╫──╫─┤M├───────────────»
#           │           │ ░  ║  ║ └╥┘┌─┐               ░  ║  ║ └╥┘┌─┐            »
#    q61_3: ┤3          ├─░──╫──╫──╫─┤M├───────────────░──╫──╫──╫─┤M├────────────»
#           └───────────┘ ░  ║  ║  ║ └╥┘ ┌───────────┐ ░  ║  ║  ║ └╥┘┌─┐         »
#    q61_4: ──────────────░──╫──╫──╫──╫──┤0          ├─░──╫──╫──╫──╫─┤M├─────────»
#                         ░  ║  ║  ║  ║  │           │ ░  ║  ║  ║  ║ └╥┘┌─┐      »
#    q61_5: ──────────────░──╫──╫──╫──╫──┤1          ├─░──╫──╫──╫──╫──╫─┤M├──────»
#                         ░  ║  ║  ║  ║  │  sub_circ │ ░  ║  ║  ║  ║  ║ └╥┘┌─┐   »
#    q61_6: ──────────────░──╫──╫──╫──╫──┤2          ├─░──╫──╫──╫──╫──╫──╫─┤M├───»
#                         ░  ║  ║  ║  ║  │           │ ░  ║  ║  ║  ║  ║  ║ └╥┘┌─┐»
#    q61_7: ──────────────░──╫──╫──╫──╫──┤3          ├─░──╫──╫──╫──╫──╫──╫──╫─┤M├»
#                         ░  ║  ║  ║  ║  └───────────┘ ░  ║  ║  ║  ║  ║  ║  ║ └╥┘»
#    q61_8: ──────────────░──╫──╫──╫──╫────────────────░──╫──╫──╫──╫──╫──╫──╫──╫─»
#                         ░  ║  ║  ║  ║                ░  ║  ║  ║  ║  ║  ║  ║  ║ »
#    q61_9: ──────────────░──╫──╫──╫──╫────────────────░──╫──╫──╫──╫──╫──╫──╫──╫─»
#                         ░  ║  ║  ║  ║                ░  ║  ║  ║  ║  ║  ║  ║  ║ »
#   q61_10: ──────────────░──╫──╫──╫──╫────────────────░──╫──╫──╫──╫──╫──╫──╫──╫─»
#                         ░  ║  ║  ║  ║                ░  ║  ║  ║  ║  ║  ║  ║  ║ »
#   q61_11: ──────────────░──╫──╫──╫──╫────────────────░──╫──╫──╫──╫──╫──╫──╫──╫─»
#                         ░  ║  ║  ║  ║                ░  ║  ║  ║  ║  ║  ║  ║  ║ »
#  meas: 12/═════════════════╩══╩══╩══╩═══════════════════╬══╬══╬══╬══╬══╬══╬══╬═»
#                            0  1  2  3                   ║  ║  ║  ║  ║  ║  ║  ║ »
#meas13: 12/══════════════════════════════════════════════╩══╩══╩══╩══╩══╩══╩══╩═»
#                                                         0  1  2  3  4  5  6  7 »
#meas14: 12/═════════════════════════════════════════════════════════════════════»
#                                                                                »
#«                         ░ ┌─┐                                 
#«    q61_0: ──────────────░─┤M├─────────────────────────────────
#«                         ░ └╥┘┌─┐                              
#«    q61_1: ──────────────░──╫─┤M├──────────────────────────────
#«                         ░  ║ └╥┘┌─┐                           
#«    q61_2: ──────────────░──╫──╫─┤M├───────────────────────────
#«                         ░  ║  ║ └╥┘┌─┐                        
#«    q61_3: ──────────────░──╫──╫──╫─┤M├────────────────────────
#«                         ░  ║  ║  ║ └╥┘┌─┐                     
#«    q61_4: ──────────────░──╫──╫──╫──╫─┤M├─────────────────────
#«                         ░  ║  ║  ║  ║ └╥┘┌─┐                  
#«    q61_5: ──────────────░──╫──╫──╫──╫──╫─┤M├──────────────────
#«                         ░  ║  ║  ║  ║  ║ └╥┘┌─┐               
#«    q61_6: ──────────────░──╫──╫──╫──╫──╫──╫─┤M├───────────────
#«                         ░  ║  ║  ║  ║  ║  ║ └╥┘┌─┐            
#«    q61_7: ──────────────░──╫──╫──╫──╫──╫──╫──╫─┤M├────────────
#«           ┌───────────┐ ░  ║  ║  ║  ║  ║  ║  ║ └╥┘┌─┐         
#«    q61_8: ┤0          ├─░──╫──╫──╫──╫──╫──╫──╫──╫─┤M├─────────
#«           │           │ ░  ║  ║  ║  ║  ║  ║  ║  ║ └╥┘┌─┐      
#«    q61_9: ┤1          ├─░──╫──╫──╫──╫──╫──╫──╫──╫──╫─┤M├──────
#«           │  sub_circ │ ░  ║  ║  ║  ║  ║  ║  ║  ║  ║ └╥┘┌─┐   
#«   q61_10: ┤2          ├─░──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫─┤M├───
#«           │           │ ░  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║ └╥┘┌─┐
#«   q61_11: ┤3          ├─░──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫──╫─┤M├
#«           └───────────┘ ░  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║ └╥┘
#«  meas: 12/═════════════════╬══╬══╬══╬══╬══╬══╬══╬══╬══╬══╬══╬═
#«                            ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║ 
#«meas13: 12/═════════════════╬══╬══╬══╬══╬══╬══╬══╬══╬══╬══╬══╬═
#«                            ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║  ║ 
#«meas14: 12/═════════════════╩══╩══╩══╩══╩══╩══╩══╩══╩══╩══╩══╩═
#«                            0  1  2  3  4  5  6  7  8  9  10 11

backend = Aer.get_backend("qasm_simulator")
shots = 1000
results = execute(sub_c, backend, shots=shots).result()
counts = results.get_counts(sub_c)
#output_distr = get_probability_distribution(counts)
plot_histogram(counts)

    # qc = get_var_form(params)
    # # Execute the quantum circuit to obtain the probability distribution associated with the current parameters
    # result = execute(qc, backend, shots=NUM_SHOTS).result()
    # # Obtain the counts for each measured state, and convert those counts into a probability vector
    # output_distr = get_probability_distribution(result.get_counts(qc))

qr = QuantumRegister(12)
#cr = QuantumRegister(3)
#circ = QuantumCircuit(qr,cr)
circ = QuantumCircuit(qr)
for i in [0,4,8]:
    sub_c = sub_circ([1,0,1,1], 5)
    sub_inst = sub_c.to_instruction()
    circ.append(sub_inst, qr[i:i+4])
    QuantumRegister(1)
    circ.measure_all()
circ.draw()


newlabel = label1
W = Signal
#predict label
for i in range(rows):
    t1 = 0    
    for j in range(0,Dims,4): 
        n = j//4
        if (label1[i,n] ==0) & (label2[i,n] ==0):
            newlabel[i,n]=0
        elif (label1[i,n] ==1) & (label2[i,n] ==1):
            newlabel[i,n]=1
        else:
            if n < Dims-1:
                temp = np.array(Signal[i,j:j+4], dtype = float)
                wtemp = np.array([1/4,1/4,1/4,1/4], dtype = float)
            else:
                temp = np.array(Signal[i,j:Dims], dtype = float)
                wtemp = np.array(1/(Dims-j)*np.ones((np.shape(Signal[i,j:Dims]))), dtype = float)                
            dw = temp*(1-2*label1[i,n])
            for steps in range(100):    
                wtemp = wtemp-dw 
                if np.mean(abs(np.ones(np.shape(wtemp))/(1+np.exp(-wtemp*temp))-label1[i,n]*label2[i,n]))<tol: #assume label 0,qubit 0
                    seq = abs(np.array(np.ones(np.shape(wtemp))/(1+np.exp(wtemp*temp)),dtype = int))
                    output = circ2(seq, 100, 0, wtemp, 4, theta, np.max(temp), np.pi/2)
    #                output1 = circ2(seq, shots, t1, wtempt)#output2 = circ1(seq, shots, t1, wtempt)
                    newlabel[i, n] = output[4]  
                    if n < Dims-1:
                        W[i,n:n+4] = wtemp
                    else:
                        W[i,n:Dims] = wtemp 
                    t1 += output[2]
                    break
                dw = temp*np.ones(np.shape(wtemp))/(1+np.exp(-wtemp*temp))
                if steps == 99:
                    newlabel[i,n]=0
                    if n < Dims-1:
                        W[i,n:n+4] = wtemp-dw
                    else:
                        W[i,n:Dims] = wtemp-dw
                    t1 += output[2]

residule = newlabel - label1
plt.figure()
plt.pcolor(label1)
plt.title('label1')

plt.figure()
plt.pcolor(newlabel)
plt.title('predicted label')

plt.figure()
plt.pcolor(residule)
plt.title('label residule')

plt.figure()
plt.pcolor(W)
plt.title('Weight')


#pre-compute

testsignal = Signal[0:samples,0] #labeled 1 (used for np.pi wave)
testsignal0 = Signal[0:samples,4] #labeled 0


#labeled1, rabi for sigmoid 1 , DDT2 for sigmoid 0
SIGMA = np.std(testsignal)
print(SIGMA) #>np.std(Signal[0:samples,:])
#49.61289706895971
SIGMAT = np.std(Signal[:,0])
print(SIGMAT)
#32.10308064170505
SIGMOID = W*Signal

#labeled0, ramsey
SIGMA0 = np.std(testsignal0)
print(SIGMA0) #>np.std(Signal[0:samples,:])
#27.48447996476684
SIGMAT0 = np.std(Signal[:,4])
print(SIGMAT0)
#14.959116520089532


# from qiskit.ignis.characterization.calibrations import rabi_schedules, RabiFitter

# from qiskit.pulse import DriveChannel
# from qiskit.compiler import assemble
# from qiskit.qobj.utils import MeasLevel, MeasReturnType
# from qiskit.providers.aer import PulseSimulator
# # Object for representing physical models
# from qiskit.providers.aer.pulse import PulseSystemModel
# # Mock Armonk backend
# from qiskit.test.mock.backends.armonk.fake_armonk import FakeArmonk

backend_config = backend.configuration()
assert backend_config.open_pulse, "Backend doesn't support Pulse"
dt = backend_config.dt*samples/fs
print(f"Sampling time: {dt*1e9} ns") 
#Sampling time: 0.022222222222222223 ns

# unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
backend_defaults = backend.defaults()
GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
KHz = 1.0e3 # Kilohertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds
ms = 1.0e-3 # mseconds

rr = 0
cc = 0
Fit = []
W = []
eta = 0.1
qubit = 0
#calibratiion and pre-computation
center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]        # The default frequency is given in Hz                                                                   # warning: this will change in a future release
print(f"Qubit {qubit} has an estimated frequency of {center_frequency_Hz / KHz} KHz.")
#Qubit 0 has an estimated frequency of 4974.462023322946 KHz.

scale_factor = 1/max(np.fft.fft(testsignal))
frequency_span_Hz = 40 * MHz
frequency_step_Hz = 1 * MHz
frequency_min = center_frequency_Hz - frequency_span_Hz / 2
frequency_max = center_frequency_Hz + frequency_span_Hz / 2
# Construct an np array of the frequencies for our experiment
frequencies_KHz = np.arange(frequency_min / KHz, 
                            frequency_max / KHz, 
                            frequency_step_Hz / KHz)
print(f"The sweep will go from {frequency_min / KHz} KHz to {frequency_max / KHz} KHz \
in steps of {frequency_step_Hz / KHz} KHz.")
#The sweep will go from 4954462.023322946 KHz to 4994462.023322946 KHz in steps of 1000.0 KHz.

from qiskit import pulse            # This is where we access all of our Pulse features!
from qiskit.pulse import Play
from qiskit.pulse import pulse_lib  # This Pulse module helps us build sampled pulses for common pulse shapes


# Drive pulse parameters (us = microseconds)
drive_sigma_us = 1/(fs*KHz)/us #0.075                     # This determines the actual width of the gaussian
drive_samples_us = drive_sigma_us*8        # This is a truncating parameter, because gaussians don't have 
                                           # a natural finite length

drive_sigma = get_closest_multiple_of_16(drive_sigma_us * us /dt)       # The width of the gaussian in units of dt
drive_samples = get_closest_multiple_of_16(drive_samples_us * us /dt)   # The truncating parameter in units of dt
drive_amp = 0.7#....
# Drive pulse samples
drive_pulse = pulse_lib.gaussian(duration=drive_samples,
                                 sigma=drive_sigma,
                                 amp=drive_amp,
                                 name='freq_sweep_excitation_pulse')

meas_map_idx = None
for i, measure_group in enumerate(backend_config.meas_map):
    if qubit in measure_group:
        meas_map_idx = i
        break
assert meas_map_idx is not None, f"Couldn't find qubit {qubit} in the meas_map!"

inst_sched_map = backend_defaults.instruction_schedule_map
measure = inst_sched_map.get('measure', qubits=backend_config.meas_map[meas_map_idx])
### Collect the necessary channels
drive_chan = pulse.DriveChannel(qubit)
meas_chan = pulse.MeasureChannel(qubit)
acq_chan = pulse.AcquireChannel(qubit)
schedule = pulse.Schedule(name='Frequency sweep')
schedule += Play(drive_pulse, drive_chan)
# The left shift `<<` is special syntax meaning to shift the start time of the schedule by some duration
schedule += measure << schedule.duration
# Create the frequency settings for the sweep (MUST BE IN HZ)
frequencies_Hz = frequencies_KHz*KHz
schedule_frequencies = [{drive_chan: freq} for freq in frequencies_Hz]

schedule.draw(label=True)

num_shots_per_frequency = samples
frequency_sweep_program = assemble(schedule,
                                   backend=backend, 
                                   meas_level=1,
                                   meas_return='avg',
                                   shots=num_shots_per_frequency,
                                   schedule_los=schedule_frequencies)
#job = backend.run(frequency_sweep_program)
#job_monitor(job)
#frequency_sweep_results = job.result(timeout=120) # timeout parameter set to 120 seconds

  


fit_params, y_fit = fit_function(frequencies_KHz,
                                 np.real(sweep_values), 
                                 lambda x, A, q_freq, B, C: (A / np.pi) * (B / ((x - q_freq)**2 + B**2)) + C,
                                 [-5, 4.975, 1, 3] # initial parameters for curve_fit
                                )

#rabi_points
# This experiment uses these values from the previous experiment:
    # `qubit`,
    # `measure`, and
    # `rough_qubit_frequency`.

# Rabi experiment parameters
num_rabi_points = samples

# Drive amplitude values to iterate over: 50 amplitudes evenly spaced from 0 to 0.75
drive_amp_min = 0
drive_amp_max = 0.85
drive_amps = np.linspace(drive_amp_min, drive_amp_max, num_rabi_points)

# Build the Rabi experiments:
#    A drive pulse at the qubit frequency, followed by a measurement,
#    where we vary the drive amplitude each time.
rabi_schedules = []
for drive_amp in drive_amps:
    rabi_pulse = pulse_lib.gaussian(duration=drive_samples, amp=drive_amp, 
                                    sigma=drive_sigma, name=f"Rabi drive amplitude = {drive_amp}")
    this_schedule = pulse.Schedule(name=f"Rabi drive amplitude = {drive_amp}")
    this_schedule += Play(rabi_pulse, drive_chan)
    # Reuse the measure instruction from the frequency sweep experiment
    this_schedule += measure << this_schedule.duration
    rabi_schedules.append(this_schedule)

rabi_schedules[-1].draw(label=True)

# Assemble the schedules into a Qobj
num_shots_per_point = samples

rabi_experiment_program = assemble(rabi_schedules,
                                   backend=backend,
                                   meas_level=1,
                                   meas_return='avg',
                                   shots=num_shots_per_point,
                                   schedule_los=[{drive_chan: rough_qubit_frequency}]
                                                * num_rabi_points)

rabi_values = []
for i in range(num_rabi_points):
    # Get the results for `qubit` from the ith experiment
    rabi_values.append(rabi_results.get_memory(i)[qubit]*scale_factor)

rabi_values = np.real(baseline_remove(rabi_values))

plt.xlabel("Drive amp [a.u.]")
plt.ylabel("Measured signal [a.u.]")
plt.scatter(drive_amps, rabi_values, color='black') # plot real part of Rabi values
plt.show()





#test   
for i in range(raws):
    if (label1[i,j//4] ==0) & (label2[i,j//4] ==0):
            newlabel.append(0)
            if (max(tsignal[i:i+samples, j+rr])-min(tsignal[i:i+samples, j+rr]))>0.1*np.max((tsignal[i:i+samples, j+rr])):  
                # rabi/ramsey
                # Rabi experiment parameters
                num_rabi_points = samples              
                # Drive amplitude values to iterate over: 50 amplitudes evenly spaced from 0 to 0.75
                drive_amp_min = 0
                drive_amp_max = 0.75
                drive_amps = np.linspace(drive_amp_min, drive_amp_max, num_rabi_points)
        rabi_schedules = []
        # Assemble the schedules into a Qobj
        num_shots_per_point = samples        
        rabi_experiment_program = assemble(rabi_schedules,
                                           backend=backend,
                                           meas_level=1,
                                           meas_return='avg',
                                           shots=num_shots_per_point,
                                           schedule_los=[{drive_chan: rough_qubit_frequency}]
        rabi_results = job.result(timeout=120)  
        rabi_values = []
        for i in range(num_rabi_points):
            # Get the results for `qubit` from the ith experiment
            rabi_values.append(rabi_results.get_memory(i)[qubit]*scale_factor)
        
        rabi_values = np.real(baseline_remove(rabi_values))
        
        plt.xlabel("Drive amp [a.u.]")
        plt.ylabel("Measured signal [a.u.]")
        plt.scatter(drive_amps, rabi_values, color='black') # plot real part of Rabi values
        plt.show() 
        fit_params0, y_fit0 = fit_function(drive_amps,
                                         rabi_values, 
                                         lambda x, A, B, drive_period, phi: (A*np.cos(2*np.pi*x/drive_period - phi) + B),
                                         [3, 0.1, 0.5, 0])
        
        plt.scatter(drive_amps, rabi_values, color='black')
        plt.plot(drive_amps, y_fit0, color='red')
        
        drive_period = fit_params0[2] # get period of rabi oscillation       
        plt.axvline(drive_period/2, color='red', linestyle='--')
        plt.axvline(drive_period, color='red', linestyle='--')
        plt.annotate("", xy=(drive_period, 0), xytext=(drive_period/2,0), arrowprops=dict(arrowstyle="<->", color='red'))
        plt.annotate("$\pi$", xy=(drive_period/2-0.03, 0.1), color='red')
        
        plt.xlabel("Drive amp [a.u.]", fontsize=15)
        plt.ylabel("Measured signal [a.u.]", fontsize=15)
        plt.show()  
        Fit.append(y_fit0)

        for drive_amp in drive_amps:
            rabi_pulse = pulse_lib.gaussian(duration=drive_samples, amp=drive_amp, 
                                            sigma=drive_sigma, name=f"Rabi drive amplitude = {drive_amp}")
            this_schedule = pulse.Schedule(name=f"Rabi drive amplitude = {drive_amp}")
            this_schedule += Play(rabi_pulse, drive_chan)
            # Reuse the measure instruction from the frequency sweep experiment
            this_schedule += measure << this_schedule.duration
            rabi_schedules.append(this_schedule)        
        for steps in range(50):
            dw = temp*(1-2*label1[i,j])
            wtemp = wtemp-dw 
            if np.mean(abs(np.ones(np.shape(wtemp))/(1+np.exp(wtemp*temp))-label1[i,j]))<tol:
                output1 = circ1(temp, shots, t1, wtempt)
                newlabel.append(output[-1])   
                W.append(wtemp)         
        pi_amp = abs(drive_period / 2)
        print(f"Pi Amplitude = {pi_amp}")
        Fts, pts = f_oneway(testsignal, tsignal[i:i+samples,j])
        Fts0, pts0 = f_oneway(testsignal0, tsignal0[i:i+samples,j])    
        pi_pulse = pulse_lib.gaussian(duration=drive_samples,
                              amp=pi_amp, 
                              sigma=drive_sigma,
                              name='pi_pulse')
        if label1[i,j//4] ==1 & label2[i,j//4] ==1:
            newlabel.append(1)
            job = backend.run(frequency_sweep_program)
            job_monitor(job)
            frequency_sweep_results = job.result(timeout=120) # timeout parameter set to 120 seconds
            sweep_values = []
            for i in range(len(frequency_sweep_results.results)):
                # Get the results from the ith experiment
                res = frequency_sweep_results.get_memory(i)*scale_factor
                # Get the results for `qubit` from this experiment
                sweep_values.append(res[qubit])
            from scipy.optimize import curve_fit
            fit_params, y_fit = fit_function(frequencies_GHz,
                                 np.real(sweep_values), 
                                 lambda x, A, q_freq, B, C: (A / np.pi) * (B / ((x - q_freq)**2 + B**2)) + C,
                                 [-5, 4.975, 1, 3] # initial parameters for curve_fit
                                )
            plt.scatter(frequencies_GHz, np.real(sweep_values), color='black') # plot real part of sweep values
            plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
            plt.xlabel("Frequency [GHz]")
            plt.ylabel("Measured signal [a.u.]")
            plt.show()
        if  label1[i,j//4]*label2[i,j//4] ==0 & label1[i,j//4]+label2[i,j//4] ==1:
            output = circ1(da, shots, t1, weight)
            newlabel.append():

    while j+nn < Dims:
        if pts < 0.05:
            testsignal =tsignal[:,j+nn]
        if pts0 < 0.05:
            testsignal0 = tsignal[:,j+nn]
           
            duration = testsignal[i:i+samples]
            scale_factor = instantaneous_frequency[0:samples,j]/center_frequency_Hz #1e-14
            
            # We will sweep 40 MHz around the estimated frequency
            frequency_span_Hz = 40000 * KHz
            # in steps of 1 MHz.
            frequency_step_Hz = 1000 * KHz
            
            # We will sweep 20000 KHz above and 20000 KHz below the estimated frequency
            frequency_min = center_frequency_Hz - frequency_span_Hz / 2
            frequency_max = center_frequency_Hz + frequency_span_Hz / 2
            # Construct an np array of the frequencies for our experiment
            frequencies_KHz = np.arange(frequency_min / KHz, 
                                        frequency_max / KHz, 
                                        frequency_step_Hz / KHz)
            
            print(f"The sweep will go from {frequency_min / KHz} KHz to {frequency_max / KHz} KHz \
                  in steps of {frequency_step_Hz / KHz} KHz.")
            #The sweep will go from 4954462.023322946 KHz to 4994462.023322946 KHz       in steps of 1000.0 KHz.
            
            # Drive pulse parameters (ms = microseconds)
            drive_sigma_ms = samples/fs      #0.1ms               # This determines the actual width of the gaussian
            drive_samples_ms = drive_sigma_ms*8        # This is a truncating parameter, because gaussians don't have 
                                                       # a natural finite length
            
            drive_sigma = get_closest_multiple_of_16(drive_sigma_ms * ms /dt)       # The width of the gaussian in units of dt
            drive_samples = get_closest_multiple_of_16(drive_samples_ms * ms /dt)   # The truncating parameter in units of dt
            drive_amp = 0.3
            # Drive pulse samples
            drive_pulse = pulse_lib.gaussian(duration=len(duration),
                                             sigma=drive_sigma,
                                             amp=drive_amp,
                                             name='freq_sweep_excitation_pulse')
            # Find out which group of qubits need to be acquired with this qubit
            meas_map_idx = None
            for i, measure_group in enumerate(backend_config.meas_map):
                if qubit in measure_group:
                    meas_map_idx = i
                    break
            assert meas_map_idx is not None, f"Couldn't find qubit {qubit} in the meas_map!"
            print(meas_map_idx)
            inst_sched_map = backend_defaults.instruction_schedule_map
            measure = inst_sched_map.get('measure', qubits=backend_config.meas_map[meas_map_idx])        
            ### Collect the necessary channels
            drive_chan = pulse.DriveChannel(qubit)
            meas_chan = pulse.MeasureChannel(qubit)
            acq_chan = pulse.AcquireChannel(qubit)
            # Create the base schedule
            # Start with drive pulse acting on the drive channel
            schedule = pulse.Schedule(name='Frequency sweep')
            schedule += Play(drive_pulse, drive_chan)
            # The left shift `<<` is special syntax meaning to shift the start time of the schedule by some duration
            schedule += measure << schedule.duration
            
            # Create the frequency settings for the sweep (MUST BE IN HZ)
            frequencies_Hz = frequencies_KHz*KHz
            schedule_frequencies = [{drive_chan: freq} for freq in frequencies_Hz]
            schedule.draw(label=True)
    
            num_shots_per_frequency = fs
            frequency_sweep_program = assemble(schedule,
                                               backend=backend, 
                                               meas_level=1,
                                               meas_return='avg',
                                               shots=num_shots_per_frequency,
                                               schedule_los=schedule_frequencies)
     #       job = backend.run(frequency_sweep_program)
            # print(job.job_id())
     #       from qiskit.tools.monitor import job_monitor
     #       job_monitor(job)
     #       frequency_sweep_results = job.result(timeout=10) # timeout parameter set to 120 seconds
            fit_params1, y_fit1 = fit_function(frequencies_KHz,
                                     np.real(duration[0:len(frequencies_KHz)]), # np.real(sweep_values),
                                     lambda x, A, q_freq, B, C: (A / np.pi) * (B / ((x - q_freq)**2 + B**2)) + C,
                                     [-5, 4.975, 1, 3] # initial parameters for curve_fit
                                    )
            Fit.append(y_fit1)
            A, rough_qubit_frequency, B, C = fit_params1
            rough_qubit_frequency = rough_qubit_frequency*KHz # make sure qubit freq is in Hz
            sc = np.mean(abs(scale_factor))
            print(f"We've updated our qubit frequency estimate from "
                  f"{round(backend_defaults.qubit_freq_est[qubit] /KHz, 5)} KHz to {round(rough_qubit_frequency/KHz/sc, 5)} KHz.")
            #We've updated our qubit frequency estimate from 4974462.02332 KHz to 52991288.18922 KHz.
            # Rabi experiment parameters
            num_rabi_points = len(duration) #len(frequencies_KHz)
            # Drive amplitude values to iterate over: 50 amplitudes evenly spaced from 0 to 0.75
            drive_amp_min = 0
            drive_amp_max = 0.75
            drive_amps = np.linspace(drive_amp_min, drive_amp_max, num_rabi_points)
            rabi_schedules = []
            
            for drive_amp in drive_amps:
                rabi_pulse = pulse_lib.gaussian(duration=num_rabi_points, amp=drive_amp, 
                                                sigma=drive_sigma, name=f"Rabi drive amplitude = {drive_amp}")
                this_schedule = pulse.Schedule(name=f"Rabi drive amplitude = {drive_amp}")
                this_schedule += Play(rabi_pulse, drive_chan)
                # Reuse the measure instruction from the frequency sweep experiment
                this_schedule += measure << this_schedule.duration
                rabi_schedules.append(this_schedule)
            
            rabi_schedules[-1].draw(label=True)
            # Assemble the schedules i#print(job.job_id())
    #job = backend.run(rabi_experiment_program)
    #job_monitor(job)nto a Qobj
            num_shots_per_point = num_rabi_points      
            rabi_experiment_program = assemble(rabi_schedules,
                                               backend=backend,
                                               meas_level=1,
                                               meas_return='avg',
                                               shots=num_shots_per_point,
                                               schedule_los=[{drive_chan: rough_qubit_frequency*1e6}]* num_rabi_points)   
            rabi_values = []
            for i in range(num_rabi_points):
                # Get the results for `qubit` from the ith experiment
                rabi_values.append(rabi_schedules[i])#rabi_results.get_memory(i)[qubit]*sc)
            
            # rabi_values = np.real(np.array(rabi_values,dtype = float)-np.mean(rabi_values)) #baseline_remove(rabi_values))
            
            # plt.xlabel("Drive amp [a.u.]")
            # plt.ylabel("Measured signal [a.u.]")
            # plt.scatter(drive_amps, rabi_values, color='black') # plot real part of Rabi values
            # plt.show()
            
            fit_params1, y_fit1 = fit_function(drive_amps,
                                     duration, 
                                     lambda x, A, B, drive_period, phi: (A*np.cos(2*np.pi*x/drive_period - phi) + B),
                                     [3, 0.1, 0.5, 0])
    
            plt.scatter(drive_amps, duration, color='black')
            plt.plot(drive_amps, y_fit1, color='red')
            drive_period = fit_params1[2] # get period of rabi oscillation        
            plt.axvline(drive_period/2, color='red', linestyle='--')
            plt.axvline(drive_period/samples, color='red', linestyle='--')
            plt.annotate("", xy=(drive_period, 0), xytext=(drive_period/2,0), arrowprops=dict(arrowstyle="<->", color='red'))
            plt.annotate("$\pi$", xy=(drive_period/2-0.03, 0.1), color='red')        
            plt.xlabel("Drive amp [a.u.]", fontsize=15)
            plt.ylabel("Measured signal [a.u.]", fontsize=15)
            plt.show()
            
            pi_amp = abs(drive_period / 2)
            print(f"Pi Amplitude = {pi_amp}")        
            #Pi Amplitude = 0.6103659963914146
            pi_pulse = pulse_lib.gaussian(duration=len(duration),
                                  amp=pi_amp, 
                                  sigma=drive_sigma,
                                  name='pi_pulse')
            # Create two schedules
            # Ground state schedule
            gnd_schedule = pulse.Schedule(name="ground state")
            gnd_schedule += measure        
            # Excited state schedule
            exc_schedule = pulse.Schedule(name="excited state")
            exc_schedule += Play(pi_pulse, drive_chan)  # We found this in Part 2A above
            exc_schedule += measure << exc_schedule.duration
            gnd_schedule.draw(label=True)
            exc_schedule.draw(label=True)
    
            # Execution settings
            num_shots = samples       
            gnd_exc_program = assemble([gnd_schedule, exc_schedule],
                                       backend=backend,
                                       meas_level=1,
                                       meas_return='single',
                                       shots=num_shots,
                                       schedule_los=[{drive_chan: rough_qubit_frequency*1e6}] * 2)
            # print(job.job_id())
            #job = backend.run(gnd_exc_program)
            #job_monitor(job)
            
            #off  resonance time dt, pi/2 (determine coherence)
            if i == rows - 96 - 1:
                duration = testsignal[i:rows]
                time_max_ms = 0.15
                time_step_ms = time_max_ms/97
                times_ms = np.arange(0.1, time_max_ms, time_step_ms)
                # Convert to units of dt
                delay_times_dt = times_ms *(dt/ms) 
                drive_amp = pi_amp / 2
                # x_90 is a concise way to say pi_over_2; i.e., an X rotation of 90 degrees
                x90_pulse = pulse_lib.gaussian(duration=len(duration),
                                               amp=drive_amp, 
                                               sigma=drive_sigma,
                                               name='x90_pulse')
                # create schedules for Ramsey experiment 
                ramsey_schedules = []   
                for delay in delay_times_dt[0:len(duration)]:
                    this_schedule = pulse.Schedule(name=f"Ramsey delay = {delay * dt / us} us")
                    this_schedule |= Play(x90_pulse, drive_chan)
                    this_schedule |= Play(x90_pulse, drive_chan) << int(this_schedule.duration + delay)
                    this_schedule |= measure << int(this_schedule.duration)           
                    ramsey_schedules.append(this_schedule)           
                ramsey_schedules[0].draw(label=True)            
                # Execution settings
                num_shots = 96
                detuning_KHz = fs 
                ramsey_frequency = round(rough_qubit_frequency*1e6 + detuning_KHz * KHz, 6) # need ramsey freq in Hz
                ramsey_program = assemble(ramsey_schedules,
                                             backend=backend,
                                             meas_level=1,
                                             meas_return='avg',
                                             shots=num_shots,
                                             schedule_los=[{drive_chan: ramsey_frequency}]*len(ramsey_schedules)
                                            )
                #job
                ramsey_values = []
                for i in range(min(len(testsignal),len(times_ms))):
                    ramsey_values.append(duration[i]+np.random.normal(1)/10)  #ramsey_results.get_memory(i)[qubit]*scale_factor)              
                plt.scatter(range(min(len(duration),len(times_ms))), np.real(ramsey_values), color='black')
                plt.xlim(0, np.max(min(len(duration),len(times_ms))))
                plt.title("Ramsey Experiment", fontsize=15)
                plt.xlabel('Delay between X90 pulses [$\mu$s]', fontsize=15)
                plt.ylabel('Measured Signal [a.u.]', fontsize=15)
                plt.show()
                
                fit_params2, y_fit2 = fit_function(times_ms, np.real(ramsey_values),
                                                 lambda x, A, del_f_KHz, C, B: (
                                                          A * np.cos(2*np.pi*del_f_KHz*x - C) + B
                                                         ),
                                                 [5, 1./0.4, 0, 0.25]
                                                )             
                # Off-resonance component
                _, del_f_KHz, _, _, = fit_params2 # freq is KHz since times in ms
                
                plt.scatter(times_ms, np.real(ramsey_values), color='black')
                plt.plot(times_ms, y_fit2, color='red', label=f"df = {del_f_KHz:.2f} KHz")
                plt.xlim(min(times_ms)-0.5*abs(min(times_ms)), max(times_ms)+0.5*abs(max(times_ms)))
                plt.xlabel('Delay between X90 pulses [$\mu$s]', fontsize=15)
                plt.ylabel('Measured Signal [a.u.]', fontsize=15)
                plt.title('Ramsey Experiment', fontsize=15)
                plt.legend()
                plt.show() 
                
                if 
            
        
x2 = Signal
x = Signal

for i in range(Dims):
    temp = Signal[:,i]
    x2[0,i] = np.var(temp)
    x[0,i] = np.mean(temp)
    for j in range(1,rows-1):        
        x2[j,i] = np.var(temp[0:j]**2)-np.var(np.mean(temp[0:j])**2)
        x[j,i] = np.mean(temp[0:j])



alpha = np.sqrt(2*m/hbar*x2)
dalpha = alpha
velocity = alpha
dalpha[0:(np.shape(alpha)[0]-1),:] = alpha[1:np.shape(alpha)[0],:]-alpha[0:(np.shape(alpha)[0]-1),:]
velocity[0:(np.shape(alpha)[0]-1),:] = x2[1:np.shape(x2)[0],:]-x2[0:(np.shape(x2)[0]-1),:]
NN = np.sqrt(np.sqrt(m/np.pi/hbar))*np.sqrt(1/wavelength) 
K = m/2/hbar/velocity
K[np.isnan(K)] = 0
K[velocity == 0] = 100* m/2/hbar
V = K
xx = K
phi = K
H = K
Z = K

#simulate dynamically
for w0 in np.linspace(2*np.pi/T-0.1*np.pi/T,2*np.pi/T+0.1*np.pi/T,2*int(np.pi/T)):
    for i in range(0,rows-96,bins):  
        for j in range(0,Dims):
          #  Omega = np.sqrt((w-w0)**2+w1**2)   do not consider coresonant first         
            datatemp = Signal[i:i+bins,j]
            b0 = hbar/2/m/datatemp
            b0[datatemp == 0] = 100*hbar/2/m  
            Pop[i:i+bins,j] = operatorP(hbar, datatemp)
            a[i:i+bins,j] = operatorA(hbar, m, np.reshape(datatemp,np.shape(Pop)), w0, Pop)
            aD[i:i+bins,j] = operatorDA(hbar, m, np.reshape(datatemp,np.shape(Pop)), w0, Pop)
            H[i:i+bins,j] = hbar*w0*(a[i:i+bins,j]*aD[i:i+bins,j]+0.5)                    
            x2 = np.zeros((np.shape(datatemp)))
            x2cor = x2
            x = x2
            aa[i:i+bins,j] = a[i:i+bins,j]
            aaD[i:i+bins,j] = aD[i:i+bins,j]
            if np.std(b0) < 0.05:
        #        if np.std(b0) < 0.05：
                if np.mean(b0) != w0:
                    coef = np.array(np.reshape((b0**2-w0**2)/w0/w0,(bins,1)),dtype = float)
                    sin2 = np.sin(w0*np.array(np.reshape(range(i,i+bins),(bins,1)),dtype = float))**2
                    assert(np.shape(coef) == np.shape(sin2))
                    post = 1+coef*sin2
                    temp = x2[i:i+bins,j]*post
                    x2cor[i:i+bins,j] = temp             
                    xx[i:i+bins,j] = np.argmax(dataset[i:i+bins,j],0)
                    V[i:i+bins,j] = m/2*w0**2*x2[i:i+bins,j] 
                    phi[i:i+bins,j] = np.sqrt(V[i:i+bins,j])
            else:
                phi[i:i+bins,j] = NN*np.exp(1j*(x2+Pop*x/2/hbar+Pop*np.sqrt(x2)/hbar+Pop*x/2/hbar)) 
                aa[i:i+bins,j] = operatorA(hbar, m, phi[i:i+bins,j], w0, Pop[i:i+bins,j])
                aaD[i:i+bins,j] = operatorDA(hbar, m, phi[i:i+bins,j], w0, Pop[i:i+bins,j])
                V[i:i+bins,j] = hbar/m*(aaD[i:i+bins,j]*aaD[i:i+bins,j]+1/2)     
                H[i:i+bins,j] = hbar*w0*(aa[i:i+bins,j]*aaD[i:i+bins,j]+0.5)
            z2[i:i+bins,j] = H[i:rows,j]*m/hbar 
# if i == rows-96:  
#     for j in range(0,Dims-1,4):
#         datatemp = dataset[i:rows,j:j+4]
#         b0 = hbar/2/m/datatemp
#         b0[datatemp == 0] = 100*hbar/2/m  
#         if np.std(b0) < 0.05:
# #        if np.std(b0) < 0.05：
#             if np.mean(b0) != w0:
#                 coef = (b0**2-w0**2)/w0/w0
#                 sin2 = np.sin(w0*np.array(np.reshape(np.repeat(range(i,rows),4),(T,4)),dtype = float))**2
#                 assert(np.shape(coef) == np.shape(sin2))
#                 post = 1+coef*sin2
#                 temp = x2[i:rows,j]*post
#                 x2cor[i:rows,j] = temp              #x2cor[i:i+T,j:j+4] = temp
#             xx[i:rows,j:j+4] = np.argmax(dataset[i:rows,j:j+4],0)
#             V[i:rows,j:j+4] = m/2*w0**2*x2[i:rows,j:j+4] 
#             phi[i:rows,j:j+4] = np.sqrt(V[i:rows,j:j+4])
#         else:
#             phi[i:rows,j:j+4] = NN*np.exp(1j*(x2+Pop*x/2/hbar+Pop*np.sqrt(x2)/hbar+Pop*x/2/hbar)) 
#             aa[i:rows,j:j+4] = operatorA(hbar, m, phi[i:rows,j:j+4], w0, Pop[i:rows,j:j+4])
#             aaD[i:rows,j:j+4] = operatorDA(hbar, m, phi[i:rows,j:j+4], w0, Pop[i:rows,j:j+4])
#             V[i:rows,j:j+4] = hbar/m*(aaD[i:rows,j:j+4]*aa[i:rows,j:j+4]+1/2)     
#     if j == Dims-1:
#         datatemp = dataset[i:rows,j]
#         b0 = hbar/2/m/datatemp
#         if np.std(b0) < 0.05:
# #        if np.std(b0) < 0.05：
#             if np.mean(b0) != w0:
#                 post = 1+(b0**2-w0**2)/w0/w0*np.sin(w0*np.array(range(i,rows),dtype = float))**2
#                 temp = x2[i:rows,j]*post
#                 x2cor[i:rows,j] = temp   #x2cor[i:i+T,j:j+4] = temp
#                 xx[i:rows,j] = np.argmax(dataset[i:rows,j],0)
#                 V[i:rows,j] = m/2*w0**2*x2[i:rows,j] 
#                 phi[i:rows,j] = np.sqrt(V[i:rows,j:j] )
#         else:
#             phi[i:rows,j] = NN*np.exp(1j*(x2+Pop*x/2/hbar+Pop*np.sqrt(x2)/hbar+Pop*x/2/hbar)) 
#             aa[i:rows,j] = operatorA(hbar, m, phi[i:rows,j], w0, Pop[i:rows,j])
#             aaD[i:rows,j] = operatorDA(hbar, m, phi[i:rows,j], w0, Pop[i:rows,j])
#             V[i:rows,j] = hbar/m*(aaD[i:i+T,j:j]*aa[i:rows,j]+1/2)               
if i == rows-96:  
    for j in range(0,Dims):
        datatemp = dataset[i:rows,j]
        b0 = hbar/2/m/datatemp
        b0[datatemp == 0] = 100*hbar/2/m  
        Pop[i:rows,j] = operatorP(hbar, datatemp)
        a[i:rows,j] = operatorA(hbar, m, np.reshape(datatemp,np.shape(Pop)), w0, Pop)
        aD[i:rows,j] = operatorDA(hbar, m, np.reshape(datatemp,np.shape(Pop)), w0, Pop)
        H[i:rows,j] = hbar*w0*(a[i:rows,j]*aD[i:rows,j]+0.5)  
        if np.std(b0) < 0.05:
#        if np.std(b0) < 0.05：
            if np.mean(b0) != w0:
                H[i:i+bins,j] = hbar*w0*(a*aD+0.5)  
                coef = (b0**2-w0**2)/w0/w0
                sin2 = np.sin(w0*np.array(np.reshape(np.repeat(range(i,rows),1),(T,1)),dtype = float))**2
                assert(np.shape(coef) == np.shape(sin2))
                post = 1+coef*sin2
                temp = x2[i:rows,j]*post
                x2cor[i:rows,j] = temp              #x2cor[i:i+T,j:j+4] = temp
            xx[i:rows,j] = np.argmax(dataset[i:rows,j],0)
            V[i:rows,j] = m/2*w0**2*x2[i:rows,j] 
            phi[i:rows,j] = np.sqrt(V[i:rows,j])
        else:
            phi[i:rows,j] = NN*np.exp(1j*(x2+Pop*x/2/hbar+Pop*np.sqrt(x2)/hbar+Pop*x/2/hbar)) 
            aa[i:rows,j] = operatorA(hbar, m, phi[i:rows,j], w0, Pop[i:rows,j])
            aaD[i:rows,j] = operatorDA(hbar, m, phi[i:rows,j], w0, Pop[i:rows,j])
            V[i:rows,j] = hbar/m*(aaD[i:rows,j]*aa[i:rows,j]+1/2)     
            H[i:rows,j] = hbar*w0*(aa[i:rows,j]*aaD[i:rows,j]+0.5)  
        z2[i:rows,j] = H[i:rows,j]*m/hbar


#theta_range = np.linspace(0, 2 * np.pi, 4)
#circuits = [qc.bind_parameters({theta: theta_val})
#            for theta_val in theta_range]

