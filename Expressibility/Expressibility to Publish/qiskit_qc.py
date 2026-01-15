import os
import warnings

import numpy as np

import qiskit
from qiskit.circuit import QuantumCircuit,Parameter, QuantumRegister, ClassicalRegister, Gate, Measure, ParameterVector
from qiskit import transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, visualize_transition, plot_bloch_vector
from qiskit.circuit.library import UnitaryGate,Initialize
from qiskit.quantum_info import Statevector,partial_trace, DensityMatrix, Operator

from typing import Union

import qutip

from scipy.sparse import block_diag, csr_matrix

from scipy.stats import unitary_group
from scipy.linalg import expm as expMatrix
import scipy.linalg
from sympy.physics.quantum.dagger import Dagger
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout': True})
colors = ['forestgreen','darkorange','dodgerblue','deeppink' ]

rng=np.random.default_rng(1)
rng2=np.random.default_rng(42)

"""

Available models: 'IQC', 'IQC_AIL', 'IQCpQ', 'IQC_Angle', and'IQCNDsE'.

"""
def av_qc():
    print("Available models: 'IQC', 'IQC_AIL', 'IQCpQ', 'IQC_Angle', and 'IQCNDsE'.")

# Normalize the dataset according to the referred model
def normalize_model(data, model=None, normalize_col=True, normalize_lin=False):
    if model==None:
        raise Exception("Input model is necessary. Available models: 'IQC', 'IQC_AIL', 'IQCpQ', 'IQC_Angle', 'IQCNDsE_Dx', and'IQCNDsE'.")
    elif model=='IQC':
        if normalize_col:
            scaler = MinMaxScaler() #Normalize the column between [0,1]
            scaler.fit(data)
            data = scaler.transform(data)
            data = data - 1
        if normalize_lin:
            data = preprocessing.normalize(data,axis=1,norm='l2') #Normalize the line between [-1,1]
    elif model=='IQC_AIL':
        if normalize_col:
            scaler = MinMaxScaler() #Normalize the column between [0,1]
            scaler.fit(data)
            data = scaler.transform(data)
            '''
            Perceba que normalizando apenas a coluna, podemos ter amplitudes dos estados em que a norma do estado não fosse igual a 1. Para resolvermos isso, devemos
            normalizar as linhas entre si

            '''
            data = preprocessing.normalize(data,axis=1,norm='l2')
        if normalize_lin:
            data = preprocessing.normalize(data,axis=1,norm='l2') #Normalize the line between [-1,1]
    else:
        if normalize_col:
            scaler = MinMaxScaler() #Normalize the column between [0,1]
            scaler.fit(data)
            data = scaler.transform(data)
        if normalize_lin:
            data = preprocessing.normalize(data,axis=1,norm='l2') #Normalize the line between [-1,1]
    
    return data

def get_weighted_sigmaQ(param,iqcpq=False):
    if iqcpq:
        n=len(param)
        diagonal=np.full(n,1)
        diagonal[-1] = -np.sum(diagonal[:-1])
        
        off_diagonal=np.full((n,n),1) - 1j*np.full((n,n),1)
        matrix=np.zeros((n,n),dtype=complex)
        np.fill_diagonal(matrix, diagonal)  # Set diagonal elements
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i, j] = off_diagonal[i, j]
                matrix[j, i] = np.conj(off_diagonal[i, j])  # Ensure Hermitian property
        return matrix
    else:
        matriz_pauli_x=np.array([[0,1],[1,0]]) # Matriz de Pauli x
        matriz_pauli_y=np.array([[0,-1j],[1j,0]]) # Matriz de Pauli y
        matriz_pauli_z=np.array([[1,0],[0,-1]]) # Matriz de Pauli z

        sigmaQ=matriz_pauli_x+matriz_pauli_y+matriz_pauli_z
        return sigmaQ

def get_U(X, vw, N_features, N_qubits, N_qubits_tgt, iqcail=False,iqcndse=False, iqcangle=False):
    

    #Montando os sigmas
    if iqcail==True:
        N_qubits_tgt=1
        X_new=np.array(X)
        w=np.array(vw)
        if np.log2(N_features)%2!=0 and np.log2(N_features)!=1:
            for k in range(2**(N_qubits-N_qubits_tgt) - N_features):
                w=np.append(vw,0)
                X_new=np.append(X_new,0)
        
        sigmaE=np.diag(w)

    elif iqcndse==True:
        X_new=np.matrix(X)
        if np.log2(N_features)%2!=0 and np.log2(N_features)!=1:
            for k in range(2**(N_qubits-N_qubits_tgt) - N_features):
                w=np.append(vw,0)
                X_new=np.append(X_new,0)
            X_new=np.matrix(X_new)
        w=np.matrix(w)
        # Ensure sigmaE is hermitian
        sigmaE = X_new.T @ w + (X_new.T @ w).T
    
    elif iqcangle==True:
        X_new=np.array(X)
        # Verifica se precisa ajustar sigmaE
        sigmaE = np.diag(w)
        # Calcula o operador unitário U
        dim_circuit = 2 ** (N_qubits - 1)
        dim_sigmaE = sigmaE.shape[0]
        sigmaE = np.kron(np.eye(dim_circuit // dim_sigmaE), sigmaE)
    
    else:
        w = np.array(vw)
        X_new=np.array(X)
        if np.log2(N_features)%2!=0 and np.log2(N_features)!=1:
            for k in range(2**(N_qubits-N_qubits_tgt) - N_features):
                w=np.append(w,0)
                X_new=np.append(X_new,0)
        sigmaE=np.diag(X_new)*w.T
    
    if N_qubits_tgt==1:
        sigma_q_params=np.full(2**N_qubits_tgt,1)
        sigmaQ=sigmaQ=get_weighted_sigmaQ(sigma_q_params,iqcpq=False)

    else:
        sigma_q_params=np.full(2**N_qubits_tgt,1)
        sigmaQ=get_weighted_sigmaQ(sigma_q_params,iqcpq=True)

    #Operador Unitário
    U=np.matrix(expMatrix(1j*np.kron(sigmaQ,sigmaE)))
    return U,X_new

# Outputs the Bloch Vector 
def blochvector(rho_cog,matriz_pauli_x,matriz_pauli_y,matriz_pauli_z):
    x_bloch = np.trace(matriz_pauli_x@rho_cog.data)
    y_bloch = np.trace(matriz_pauli_y@rho_cog.data)
    z_bloch = np.trace(matriz_pauli_z@rho_cog.data)
    return [x_bloch,y_bloch,z_bloch]
    
# Execute qiskit circuit
def run_qasm_counts(qc, shots, N_qubits_tgt, backend='qasm_simulator'):
    qc.measure([i for i in range(N_qubits_tgt)],[i for i in range(N_qubits_tgt)])
    qasm_simulator = Aer.get_backend(backend)
    job = qasm_simulator.run(qc, shots=shots)
    result = job.result()
    return result.get_counts()

# Execute qiskit circuit
def run_qasm_counts_meas_all(qc, shots):
    qc.measure_all()
    qasm_simulator = Aer.get_backend("qasm_simulator")
    job = qasm_simulator.run(qc, shots=shots)
    result = job.result()
    return result.get_counts()

# Builds up the negativity list through the referred model
def get_negativity(rho, dim):
    """
        Returns the Negativity associated with densitiy matrix rho.
        See definition at: https://en.wikipedia.org/wiki/Negativity_(quantum_mechanics)
    """
    """
        Returns the Negativity associated with densitiy matrix rho.
        See definition at: https://en.wikipedia.org/wiki/Negativity_(quantum_mechanics)
        See implementation at: https://toqito.readthedocs.io/en/latest/_autosummary/toqito.state_props.negativity.html
    
    return state_props.negativity(rho, dim)
    """
    
    d1, d2 = dim
    rho_reshaped = rho.reshape(d1, d2, d1, d2)
    pt_rho = rho_reshaped.transpose(0, 3, 2, 1).reshape(d1*d2, d1*d2)
    eigenvalues = np.linalg.eigvals(pt_rho)
    return float(np.sum(np.abs(eigenvalues[eigenvalues < 0])))


# Builds up the model NOT to calculate expressibility
def circuit_model(data, contador, w, counter, qubits, N_qubits, N_features, N_qubits_tgt=1, model=None,
                  folder=None, printar_cirq=False, transpilar=False, N_layers=None):

    if model=='IQC':
        
        # IQC
        qc = QuantumCircuit(N_qubits,N_qubits_tgt)

        qc.h(range(N_qubits))

        #Operador Unitário
        U, X_new = get_U(data, w, N_features, N_qubits, N_qubits_tgt)

        # Adiciona o operador unitário ao circuito
        qc.unitary(U,qubits)
        
        if counter==0:
            qc.draw("mpl", filename=folder+f'/mpl_complete_U_NF{N_features}_{model}.svg')
        if printar_cirq==True:
            display(qc.draw('mpl')) #display(qc.draw("mpl", filename='./mpl_original.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos

        if transpilar==True:
            #qc.decompose().draw(output="mpl", style="clifford")
            tqc=transpile(qc, optimization_level=0, basis_gates=['u3', 'x', 'h', 'z', 'cx'],seed_transpiler=1)

            gate_val = 0
            u3_dir = {}
            for i, instruction in enumerate(tqc.data):
                if instruction.operation.name == 'u3':
                    u3_dir['u3_'+str(gate_val)] = {'qubit':instruction.qubits[0], 'params': instruction.operation.params}
                    gate_val +=1
                    
            if printar_cirq and dict(tqc.count_ops())['u3']<=50:
                print(u3_dir)
                print()

            
            u3_params = []
            for i in range(len(u3_dir)):
                u3_params.append(u3_dir[f'u3_{i}']['params'])

            if dict(tqc.count_ops())['u3']<=50 and contador==0:
                tqc.draw("mpl", filename=folder+f'/mpl_transpiled{contador}_NF{N_features}_{model}.svg')

            if printar_cirq==True and dict(tqc.count_ops())['u3']<=50:
                print(dict(tqc.count_ops()))
                display(tqc.draw('mpl')) #displat(qc.draw('mpl', filename='./mpl_transpile.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos


            # Mostrando o vetor de estado 
            sv = Statevector(qc)
            rho=np.array(DensityMatrix(sv))
            rho_cog = partial_trace(sv, qubits[1:])
            if printar_cirq==True:
                print(rho_cog)
            
            return qc,u3_params, get_negativity(rho,[2**N_qubits_tgt, len(X_new)])
        else:
            # Mostrando o vetor de estado 
            sv = Statevector(qc)
            rho=np.array(DensityMatrix(sv))
            rho_cog = partial_trace(sv, qubits[1:])
            if printar_cirq==True:
                print(rho_cog)
            return qc,0, get_negativity(rho,[2**N_qubits_tgt, len(X_new)])
    
    elif model=='IQC_AIL':       

        # IQC_AIL
        qc = QuantumCircuit(N_qubits,N_qubits_tgt)
        qc.initialize(X_new, range(1,N_qubits))# Inicializaçao do estado inicial. Poderia ser qualquer estado.
        qc.h(0)

        #Operador Unitário
        U, X_new = get_U(data, w, N_features, N_qubits, N_qubits_tgt, iqcail=True)

        # Adiciona o operador unitário ao circuito
        qc.unitary(U,qubits)
        
        if counter==0:
            qc.draw("mpl", filename=folder+f'/mpl_complete_U_NF{N_features}_{model}.svg')
        if printar_cirq==True:
            display(qc.draw('mpl')) #display(qc.draw("mpl", filename='./mpl_original.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos

        if transpilar==True:
            #qc.decompose().draw(output="mpl", style="clifford")
            tqc=transpile(qc, optimization_level=0, basis_gates=['u3', 'x', 'h', 'z', 'cx'],seed_transpiler=1)

            gate_val = 0
            u3_dir = {}
            for i, instruction in enumerate(tqc.data):
                if instruction.operation.name == 'u3':
                    u3_dir['u3_'+str(gate_val)] = {'qubit':instruction.qubits[0], 'params': instruction.operation.params}
                    gate_val +=1
                    
            if printar_cirq and dict(tqc.count_ops())['u3']<=50:
                print(u3_dir)
                print()

            
            u3_params = []
            for i in range(len(u3_dir)):
                u3_params.append(u3_dir[f'u3_{i}']['params'])

            if dict(tqc.count_ops())['u3']<=50 and contador==0:
                tqc.draw("mpl", filename=folder+f'/mpl_transpiled{contador}_NF{N_features}_{model}.svg')

            if printar_cirq==True and dict(tqc.count_ops())['u3']<=50:
                print(dict(tqc.count_ops()))
                display(tqc.draw('mpl')) #displat(qc.draw('mpl', filename='./mpl_transpile.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos


            # Mostrando o vetor de estado 
            sv = Statevector(qc)
            rho=np.array(DensityMatrix(sv))
            rho_cog = partial_trace(sv, qubits[1:])
            if printar_cirq==True:
                print(rho_cog)
            
            return qc,u3_params, get_negativity(rho,[2**N_qubits_tgt, len(X_new)])
        else:
            # Mostrando o vetor de estado 
            sv = Statevector(qc)
            rho=np.array(DensityMatrix(sv))
            rho_cog = partial_trace(sv, qubits[1:])
            if printar_cirq==True:
                print(rho_cog)
            return qc,0, get_negativity(rho,[2**N_qubits_tgt, len(X_new)])
        
    elif model=='IQCpQ': # IQC Expanding psiQ
        
        N_qubits_env=N_qubits
        N_qubits_env-=N_qubits_tgt
        
        # IQCpQ
        qc = QuantumCircuit(N_qubits_env+N_qubits_tgt,N_qubits_tgt)
        qc.h(range(N_qubits_env+N_qubits_tgt))

        #Operador Unitário
        U, X_new = get_U(data, w, N_features, N_qubits, N_qubits_tgt)

        # Adiciona o operador unitário ao circuito
        qc.unitary(U,qubits)

        if counter==0:
            qc.draw("mpl", filename=folder+f'/mpl_complete_U_NF{N_features}_{model}.svg')
        if printar_cirq==True:
            display(qc.draw('mpl')) #display(qc.draw("mpl", filename='./mpl_original.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos

        if transpilar==True:
            #qc.decompose().draw(output="mpl", style="clifford")
            tqc=transpile(qc, optimization_level=0, basis_gates=['u3', 'x', 'h', 'z', 'cx'],seed_transpiler=1)

            gate_val = 0
            u3_dir = {}
            for i, instruction in enumerate(tqc.data):
                if instruction.operation.name == 'u3':
                    u3_dir['u3_'+str(gate_val)] = {'qubit':instruction.qubits[0], 'params': instruction.operation.params}
                    gate_val +=1
                    
            if printar_cirq and dict(tqc.count_ops())['u3']<=50:
                print(u3_dir)
                print()

            
            u3_params = []
            for i in range(len(u3_dir)):
                u3_params.append(u3_dir[f'u3_{i}']['params'])

            if dict(tqc.count_ops())['u3']<=50 and contador==0:
                tqc.draw("mpl", filename=folder+f'/mpl_transpiled{contador}_NF{N_features}_{model}.svg')

            if printar_cirq==True and dict(tqc.count_ops())['u3']<=50:
                print(dict(tqc.count_ops()))
                display(tqc.draw('mpl')) #displat(qc.draw('mpl', filename='./mpl_transpile.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos


            # Mostrando o vetor de estado 
            sv = Statevector(qc)
            rho=np.array(DensityMatrix(sv))
            rho_cog = partial_trace(sv, qubits[1:])
            if printar_cirq==True:
                print(rho_cog)
            
            return qc,u3_params, get_negativity(rho,[2**N_qubits_tgt, len(X_new)])
        else:
            # Mostrando o vetor de estado 
            sv = Statevector(qc)
            rho=np.array(DensityMatrix(sv))
            rho_cog = partial_trace(sv, qubits[1:])
            if printar_cirq==True:
                print(rho_cog)
            return qc,0, get_negativity(rho,[2**N_qubits_tgt, len(X_new)])
            
    elif model=='IQCNDsE': # IQC Non Diagonal sigmaE: sE=X_new.T @ w + (X_new.T @ w).T  

        # IQCNDsE
        qc = QuantumCircuit(N_qubits,N_qubits_tgt)
        qc.h(range(N_qubits))        

        #Operador Unitário
        U, X_new = get_U(data, w, N_features, N_qubits, N_qubits_tgt, iqcndse=True)

        # Adiciona o operador unitário ao circuito
        qc.unitary(U,qubits)

        if counter==0:
            qc.draw("mpl", filename=folder+f'/mpl_complete_U_NF{N_features}_{model}.svg')
        if printar_cirq==True:
            display(qc.draw('mpl')) #display(qc.draw("mpl", filename='./mpl_original.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos

        if transpilar==True:
            #qc.decompose().draw(output="mpl", style="clifford")
            tqc=transpile(qc, optimization_level=0, basis_gates=['u3', 'x', 'h', 'z', 'cx'],seed_transpiler=1)

            gate_val = 0
            u3_dir = {}
            for i, instruction in enumerate(tqc.data):
                if instruction.operation.name == 'u3':
                    u3_dir['u3_'+str(gate_val)] = {'qubit':instruction.qubits[0], 'params': instruction.operation.params}
                    gate_val +=1
                    
            if printar_cirq and dict(tqc.count_ops())['u3']<=50:
                print(u3_dir)
                print()

            
            u3_params = []
            for i in range(len(u3_dir)):
                u3_params.append(u3_dir[f'u3_{i}']['params'])

            if dict(tqc.count_ops())['u3']<=50 and contador==0:
                tqc.draw("mpl", filename=folder+f'/mpl_transpiled{contador}_NF{N_features}_{model}.svg')

            if printar_cirq==True and dict(tqc.count_ops())['u3']<=50:
                print(dict(tqc.count_ops()))
                display(tqc.draw('mpl')) #displat(qc.draw('mpl', filename='./mpl_transpile.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos


            # Mostrando o vetor de estado 
            sv = Statevector(qc)
            rho=np.array(DensityMatrix(sv))
            rho_cog = partial_trace(sv, qubits[1:])
            if printar_cirq==True:
                print(rho_cog)
            
            return qc,u3_params, get_negativity(rho,[2**N_qubits_tgt, len(X_new.T)])
        else:
            # Mostrando o vetor de estado 
            sv = Statevector(qc)
            rho=np.array(DensityMatrix(sv))
            rho_cog = partial_trace(sv, qubits[1:])
            if printar_cirq==True:
                print(rho_cog)
            return qc,0, get_negativity(rho,[2**N_qubits_tgt, len(X_new.T)])

    elif model=='IQC_Angle': # IQC with angle embedding
        
        if N_layers==None:
            raise Exception("Number of Layers is required in Angle Embedding.")
        
        N_QUBITS=(N_features+N_qubits_tgt) #Nqubits do circuito
        QUBITS=[i for i in range(N_QUBITS)]
        N_layers = N_layers

        # Configura o circuito
        qc = QuantumCircuit(N_QUBITS,N_qubits_tgt)

        # Adiciona a porta Hadamard no qubit alvo
        qc.h(0)

        # Adiciona as rotações RX e as CNOTs
        for nl in range(N_layers):
            for i in range(len(X_new)):
                if i + 1 < N_QUBITS:
                    qc.rx(X_new[i] * 2 * np.pi, i + 1)
                    if i != 0:
                        qc.cx(i, i + 1)
        
        U, X_new = get_U(data, w, N_features, N_QUBITS, N_qubits_tgt, iqcangle=True)

        # Adiciona o operador unitário ao circuito
        qc.unitary(U, QUBITS)

        if counter==0:
            qc.draw("mpl", filename=folder+f'/mpl_complete_U_NF{N_features}_{model}.svg')
        if printar_cirq==True:
            display(qc.draw('mpl')) #display(qc.draw("mpl", filename='./mpl_original.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos

        if transpilar==True:
            #qc.decompose().draw(output="mpl", style="clifford")
            tqc=transpile(qc, optimization_level=0, basis_gates=['u3', 'x', 'h', 'z', 'cx'],seed_transpiler=1)

            gate_val = 0
            u3_dir = {}
            for i, instruction in enumerate(tqc.data):
                if instruction.operation.name == 'u3':
                    u3_dir['u3_'+str(gate_val)] = {'qubit':instruction.qubits[0], 'params': instruction.operation.params}
                    gate_val +=1
                    
            if printar_cirq and dict(tqc.count_ops())['u3']<=50:
                print(u3_dir)
                print()

            
            u3_params = []
            for i in range(len(u3_dir)):
                u3_params.append(u3_dir[f'u3_{i}']['params'])

            if dict(tqc.count_ops())['u3']<=50 and contador==0:
                tqc.draw("mpl", filename=folder+f'/mpl_transpiled{contador}_NF{N_features}_{model}.svg')

            if printar_cirq==True and dict(tqc.count_ops())['u3']<=50:
                print(dict(tqc.count_ops()))
                display(tqc.draw('mpl')) #display(qc.draw('mpl', filename='./mpl_transpile.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos


            # Mostrando o vetor de estado 
            sv = Statevector(qc)
            rho=np.array(DensityMatrix(sv))
            rho_cog = partial_trace(sv, QUBITS[1:])
            if printar_cirq==True:
                print(rho_cog)

            
            return qc,u3_params, get_negativity(rho,[2**N_qubits_tgt, 2**N_features])
        else:
            # Mostrando o vetor de estado 
            sv = Statevector(qc)
            rho=np.array(DensityMatrix(sv))
            rho_cog = partial_trace(sv, QUBITS[1:])
            if printar_cirq==True:
                print(rho_cog)
            return qc,0, get_negativity(rho,[2**N_qubits_tgt, 2**N_features])
        

    elif model==None:
        raise Exception("Input model is necessary. Available models: 'IQC', 'IQC_AIL', 'IQCpQ', 'IQC_Angle', and 'IQCNDsE'.")#, and 'IQC_AIL_RU'.")
    
    if folder==None:
        raise Exception("No folder selected.")

# Define a function to check and crop possible lists with different sizes 
# Called up in plot_histogram function
def size_divide(lista):
    # Dicionário para armazenar sublistas agrupadas pelo tamanho
    grupos = {}
    
    for sublista in lista:
        tamanho = len(sublista)
        if tamanho not in grupos:
            grupos[tamanho] = []
        grupos[tamanho].append(sublista)
    
    return grupos

def esfera_bloch(X,weights,qubits,N_qubits,N_features,counter,model=None,folder=None,printar_esf=False,norma=None,N_qubits_tgt=None):
    if model==None:
        raise Exception("Input model is necessary. Available models: 'IQC', 'IQC_AIL', 'IQCpQ', 'IQCNDsE_wx', 'IQCNDsE_Dx', and 'IQCNDsE'.")#, and 'IQC_AIL_RU'.")
    
    if model=='IQCpQ' and N_qubits_tgt==None:
        raise Exception("In 'IQCpQ' model, giving 'N_qubit_tgt' is required.")
    
    if folder==None:
        raise Exception("No folder selected.")
    
    point_states=[]
    u3_params=[]
    negativity=[]
    for k in range(len(X)):
        bloch,params,neg=circuit_model(X[k],k,weights[k], counter, qubits, N_qubits, N_features,folder=folder,model=model,N_qubits_tgt=N_qubits_tgt)
        point_states.append(bloch)
        u3_params.append(params)
        negativity.append(neg)
        counter+=1


    b = qutip.Bloch()
    b.point_default_color=['k']
    b.point_marker=['o']
    b.point_size=[10, 10, 10, 10]
    for k in range(len(point_states)):
        b.add_points(point_states[k])
    b.render()
    if printar_esf==True:
        b.show()

    bb = b.fig
    if norma:
        bb.savefig(fname=folder+f'/Bloch_geral_NF{N_features}_{model}_{norma}.svg')
    else:
        bb.savefig(fname=folder+f'/Bloch_geral_NF{N_features}_{model}.svg')
    return u3_params,negativity

def plot_histogram_qc(u3_list,neg_list,N_features,folder=None,norma=None, model=None):
    
    if model==None:
        raise Exception("Input model is necessary. Available models: 'IQC', 'IQC_AIL', 'IQCpQ', 'IQCNDsE_wx', 'IQCNDsE_Dx', 'IQCNDsE', and 'IQC_AIL_RU'.")
    
    if folder==None:
        raise Exception("No folder selected.")
    
    lens=[]
    for i in range(len(u3_list)):
        lens.append(len(u3_list[i]))

    unique_G_list=np.unique(lens)

    to_binning=np.unique_counts(lens)
    bin_sizes_list=[]
    for i in range(len(to_binning[0])):
        if to_binning[1][i]<=100:
            bin_sizes_list.append(to_binning[1][i])
        elif to_binning[1][i]<=1000 and to_binning[1][i]>100:
            bin_sizes_list.append(to_binning[1][i]//10)
        elif to_binning[1][i]>1000:
            bin_sizes_list.append(to_binning[1][i]//100)

    
    lista=size_divide(u3_list) # lista[G size index][tqc index][gate index][params] --> lista[int(np.unique[j])][:][i][0 or 1 or 2]
    neg_list=np.array(neg_list)

    for j in range(len(unique_G_list)):
        for i in range(unique_G_list[j]):
            array=np.array(lista[unique_G_list[j]]) # array[tqc index,gates index,parameters index]
            fig,ax=plt.subplots(1,3,figsize=(15,5))
            ax[0].hist(array[:,i,0]/np.pi,label=labels[0],color=colors[0], bins=bin_sizes_list[j],edgecolor='black') # array[tqc index,gates index,parameters index]
            ax[0].set_xlabel(f'Factor of $\pi$')
            ax[0].set_ylabel('Frequency')
            ax[0].legend()
            ax[0].grid(linestyle='dashed')

            ax[1].hist(array[:,i,1]/np.pi,label=labels[1],color=colors[1], bins=bin_sizes_list[j],edgecolor='black') # array[tqc index,gates index,parameters index]
            ax[1].set_xlabel(f'Factor of $\pi$')
            ax[1].set_ylabel('Frequency')
            ax[1].legend()
            ax[1].grid(linestyle='dashed')

            ax[2].hist(array[:,i,2]/np.pi,label=labels[2],color=colors[2], bins=bin_sizes_list[j],edgecolor='black') # array[tqc index,gates index,parameters index]
            ax[2].set_xlabel(f'Factor of $\pi$')
            ax[2].set_ylabel('Frequency')
            ax[2].legend()
            ax[2].grid(linestyle='dashed')
            if norma: 
                """
                If you want to plot the effects of normalization on this, change 'norma' parameter of the function to 'column' or 'line".
                If you want both, do the changes just described and call it twice: one for each type of normalization.
                """
                plt.savefig(folder+f'/histogram_NF{N_features}_gate{i}_{norma}_G{unique_G_list[j]}_{model}.svg') 
            else:
                plt.savefig(folder+f'/histogram_NF{N_features}_gate{i}_G{unique_G_list[j]}_{model}.svg')
            plt.close(fig)

def plot_negativity(neg_list1,N_samples,N_features,folder=None,neg_list2=None,model=None,normalization=False):
    if folder==None:
        raise Exception("No folder selected.")

    if model=='IQC':
        model_title = 'IQC'
    elif model=='IQC_AIL':
        model_title = 'IQC_AIL'
    elif model=='IQCpQ': # IQC Expanding psiQ
        model_title = 'IQCpQ'
    elif model=='IQCNDsE_wx': # IQC Non Diagonal sigmaE: sE=w.T@x
        model_title = 'IQCNDsE_wx'
    elif model=='IQCNDsE_Dx': # IQC Non Diagonal sigmaE: x elements occupy the diagonal of sigmaE
        model_title = 'IQCNDsE_Dx'
    elif model=='IQCNDsE': # IQC Non Diagonal sigmaE: sE=x.T@w  
        model_title = 'IQCNDsE'    
    elif model=='IQC_AIL_RU': # IQC_AIL with arbitrary U operator
        model_title = 'IQC_AIL_RU'
    elif model=='IQC_RU_Dx': # IQC with features vector embedded in U operator
        model_title = 'IQC_RU_Dx'
    elif model==None:
        raise Exception("Input model is necessary. Available models: 'IQC', 'IQC_AIL', 'IQCpQ', 'IQCNDsE_wx', 'IQCNDsE_Dx', 'IQCNDsE', and 'IQC_AIL_RU'.")

    num_neg=[]
    for i in range(N_samples):
        num_neg.append(i)

    if neg_list2:
        normalization=True
    if normalization:

        fig,ax=plt.subplots(1,2,figsize=(15,5))

        ax[0].scatter(num_neg,neg_list1, marker='.', s=12)
        ax[0].set_ylabel('Negativity Value')
        ax[0].set_title('Normalized by Column')

        plt.suptitle(f'Negativity throughout the {model_title} Model with N_features = {N_features}')


        ax[1].scatter(num_neg,neg_list2, marker='.', s=12)
        ax[1].set_ylabel('Negativity Value')
        ax[1].set_title('Normalized by Line')
        plt.savefig(folder+f'/Negativity_NF{N_features}_{model}.svg')
        plt.close(fig)
    else:
        fig,ax=plt.subplots(1,1,figsize=(15,5))

        ax.scatter(num_neg,neg_list1, marker='.', s=12)
        ax.set_ylabel('Negativity Value')
        plt.savefig(folder+f'/Negativity_NF{N_features}_{model}.svg')
        plt.close(fig)

def statistical_qc(N_samples,N_features,model=None,folder=None,normalization=False,N_qubits_tgt=None,esfera=False,N_layers=None):
    if folder==None:
        raise Exception("No folder selected.")
    if model==None:
        raise Exception("Input model is necessary. Available models: 'IQC', 'IQC_AIL', 'IQCpQ', 'IQCNDsE', and 'IQC_Angle'.")#, and 'IQC_AIL_RU'.")
    if N_qubits_tgt:
        N_qubits=math.ceil(np.log2(N_features)+N_qubits_tgt)
    else:
        N_qubits=math.ceil(np.log2(N_features)+1) #Nqubits do circuito
    counter=0
    X_df=rng.random((N_samples,N_features))
    w_df=rng2.random((N_samples,N_features))
    
    qubits=[i for i in range(N_qubits)]

    if esfera==True:
        if normalization:
            X_df_coluna=normalize_model(X_df,model=model, normalize_col=True, normalize_lin=False)
            X_df_linha=normalize_model(X_df,model=model,normalize_col=False,normalize_lin=True)
            u3_col,neg_col=esfera_bloch(X_df_coluna,w_df,qubits,N_qubits,N_features,counter,model=model,folder=folder,norma='coluna',N_qubits_tgt=N_qubits_tgt,N_layers=N_layers)
            u3_lin,neg_lin=esfera_bloch(X_df_linha,w_df,qubits,N_qubits,N_features,counter,model=model,folder=folder,norma='linha',N_qubits_tgt=N_qubits_tgt,N_layers=N_layers)
            return u3_col, u3_lin, neg_col, neg_lin
        else:
            X_df=normalize_model(X_df,model=model,normalize_col=True,normalize_lin=False)
            u3_lista,neg_lista=esfera_bloch(X_df,w_df,qubits,N_qubits,N_features,counter=counter,model=model,folder=folder,N_qubits_tgt=N_qubits_tgt,N_layers=N_layers)
            return u3_lista, neg_lista
    else:
        #Calculating Negativity and U3 gates Histogram 
        u3_lista=[]
        negativity=[]
        for k in range(len(X_df)):
            _,params,neg=circuit_model(X_df[k],k,w_df[k], counter, qubits, N_qubits, N_features,folder=folder,model=model,N_qubits_tgt=N_qubits_tgt,N_layers=N_layers)
            u3_lista.append(params)
            negativity.append(neg)
            counter+=1
        return u3_lista, neg_lista

def P_harr(l,u,N):
    return (1-l)**(N-1)-(1-u)**(N-1)

def bins(N_qubits, N_bins=75):
    #Possible Bin
    bins_list=[]
    for i in range(N_bins+1):
        bins_list.append((i)/N_bins)
    #Center of the Bean
    bins_x=[]    
    for i in range(N_bins):
        bins_x.append(bins_list[1]+bins_list[i])
    
    #Harr histogram
    P_harr_hist=[]
    for i in range(N_bins):
        P_harr_hist.append(P_harr(bins_list[i],bins_list[i+1],2**(N_qubits)))    
    #Imaginary    
    #j=(-1)**(1/2)
    return P_harr_hist, bins_x, bins_list

def get_U_operator_altered(params, N_features, N_qubits, N_qubits_tgt, iqcail=False,iqcndse=False, iqcangle=False):
    X = params[:N_features]
    vw = params[N_features:]
    #Montando os sigmas
    if iqcail==True:
        N_qubits_tgt=1
        X_new=np.array(X)
        w=np.array(vw)
        if np.log2(N_features)%2!=0 and np.log2(N_features)!=1:
            for k in range(2**(N_qubits-N_qubits_tgt) - N_features):
                w=np.append(vw,0)
                X_new=np.append(X_new,0)
        
        sigmaE=np.diag(w)

    elif iqcndse==True:
        atx=np.array(X)
        atw=np.array(vw)
        if np.log2(N_features)%2!=0 and np.log2(N_features)!=1:
            for k in range(2**(N_qubits-N_qubits_tgt) - N_features):
                atw=np.append(atw,0)
                atx=np.append(atx,0)
        X_new=np.matrix(atx)
        w=np.matrix(atw)
        # Ensure sigmaE is hermitian
        sigmaE = X_new.T @ w + (X_new.T @ w).T
    
    elif iqcangle==True:
        X_new=np.array(X)
        # Verifica se precisa ajustar sigmaE
        sigmaE = vw
        # Calcula o operador unitário U
        dim_circuit = 2 ** (N_qubits-1)
        dim_sigmaE = len(sigmaE)
        #sigmaE = np.kron(np.eye(dim_circuit // dim_sigmaE), sigmaE)
        sigmaE = np.kron(np.ones(dim_circuit // dim_sigmaE), sigmaE)
        if np.log2(len(sigmaE))%2!=0 and np.log2(len(sigmaE))!=1: # Padding sigmaE
            for k in range(2**(N_qubits-N_qubits_tgt) - len(sigmaE)):
                sigmaE=np.append(sigmaE,0)

        sigmaE=np.diag(sigmaE)
    
    else:
        w = np.array(vw)
        X_new=np.array(X)
        if np.log2(N_features)%2!=0 and np.log2(N_features)!=1:
            for k in range(2**(N_qubits-N_qubits_tgt) - N_features):
                w=np.append(w,0)
                X_new=np.append(X_new,0)
        sigmaE=np.diag(X_new)*w.T
    
    if N_qubits_tgt==1:
        sigma_q_params=np.full(2**N_qubits_tgt,1)
        sigmaQ=get_weighted_sigmaQ(sigma_q_params,iqcpq=False)

    else:
        sigma_q_params=np.full(2**N_qubits_tgt,1)
        sigmaQ=get_weighted_sigmaQ(sigma_q_params,iqcpq=True)

    #Operador Unitário
    #U_exp = np.matrix(expMatrix(1j*np.kron(sigmaQ,sigmaE)))
    U = np.matrix(np.kron(np.identity(2**N_qubits_tgt),np.cos(np.sqrt(3)*sigmaE)) + (1j/np.sqrt(3))*np.kron(sigmaQ,np.sin(np.sqrt(3)*sigmaE)), dtype=complex)
    return U

"""def get_U_sparse(tx, tw):
    # Definição da matriz sigmaQ
    sigmaQ = np.array([[1, 1-1j],
                    [1+1j, 1]], dtype=complex)

    def exp_i_lambda_sigmaQ(lambda_val):
        # parâmetros fixos
        tr2 = 1.0            # tr(sigmaQ)/2
        s = np.sqrt(2.0)     # já calculado
        prefactor = np.exp(1j * lambda_val * tr2)  # e^{i lambda * tr/2} = e^{i lambda}
        c = np.cos(lambda_val * s)
        si = np.sin(lambda_val * s)
        M = prefactor * ( c * np.eye(2, dtype=complex) + (1j * si / s) * (sigmaQ - np.eye(2, dtype=complex)) )
        return M

    # exemplo: vetor de lambdas (os elementos diagonais de sigma_E)
    lambdas = np.array([tx*tw for tx,tw in zip(tx,tw)])

    # crie a lista de blocos 2x2
    blocks = [exp_i_lambda_sigmaQ(lam) for lam in lambdas]

    # escolha: construir matriz densa 2m x 2m
    U_dense = np.block([[blocks[i] if i==j else np.zeros((2,2), dtype=complex) 
                        for j in range(len(blocks))] 
                        for i in range(len(blocks))])
    blocks = [np.array(b, dtype=np.complex128) for b in blocks]
    U_sparse = block_diag(blocks, format='csr', dtype=np.complex128)
    # ou — mais eficiente em memória — construir esparso em CSR
    #U_sparse = block_diag(blocks, format='csr')

    # use U_sparse (ou U_dense) conforme necessidade
    return U_sparse, U_dense

def get_U_operator_trotterized(params, N_qubits, N_layers=2, t=1.0):
    '''
    Gera um circuito trotterizado para aproximar exp(i * kron(sigmaQ, sigmaE))
    sem construir a matriz completa.
    '''
    qc = QuantumCircuit(N_qubits)
    n_steps = 5  # número de passos de Trotter (ajustável)
    dt = t / n_steps

    # Aqui, cada sigma é substituído por rotações nos qubits correspondentes.
    # Exemplo genérico:
    for _ in range(n_steps):
        for i in range(N_qubits):
            qc.rx(2 * params[i] * dt, i)
            qc.rz(2 * params[i] * dt, i)
            qc.ry(2 * params[i] * dt, i)

    return qc
"""

def conj_reversed_qc(qc: QuantumCircuit):
    
    rev_ops = reversed(qc.data)
    U_dagger = None
    for gate, qargs, cargs in rev_ops:
        new_gate = gate
        if gate.params:
            new_params = [Parameter(f'conj_{param.name}') for param in gate.params]
            if hasattr(gate, 'N_features') and hasattr(gate, 'N_qubits_tgt'):
                # Caso especial para nosso gate personalizado
                new_gate = gate.__class__(name=gate.name,
                                        num_qubits=gate.num_qubits,
                                        params=new_params,
                                        N_features=gate.N_features,
                                        N_qubits_tgt=gate.N_qubits_tgt)
                U_dagger = new_gate
            else:
                # Para gates padrão do Qiskit
                new_gate = gate.__class__(*new_params)
        
        qc.append(new_gate, qargs, cargs)
    return qc, U_dagger

def conj_reversed_qc_angle(qc: QuantumCircuit):
    """
    Cria um circuito estendido com:
    1. O circuito original
    2. Seu reverso conjugado (com parâmetros prefixados por 'conj_')
    """
    # Cria uma cópia do circuito original
    extended_qc = qc.copy()
    
    # Dicionário para mapear parâmetros originais para conjugados
    param_map = {}
    
    # Primeira passada: identificar todos os parâmetros únicos
    for instruction in qc.data:
        gate = instruction.operation
        if hasattr(gate, 'params'):
            for param in gate.params:
                if isinstance(param, Parameter) and param.name not in param_map:
                    param_map[param] = Parameter(f'conj_{param.name}')
    
    U_dagger = None

    # Segunda passada: adicionar operações invertidas com parâmetros conjugados
    for instruction in reversed(qc.data):
        gate = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits
        
        new_gate = gate
        if hasattr(gate, 'params') and gate.params:
            # Substitui os parâmetros pelos conjugados
            new_params = [param_map.get(p, p) if isinstance(p, Parameter) else p 
                         for p in gate.params]
            
            if hasattr(gate, 'N_features') and hasattr(gate, 'N_qubits_tgt'):
                # Gate personalizado
                new_gate = gate.__class__(
                    name=gate.name,
                    num_qubits=gate.num_qubits,
                    params=new_params,
                    N_features=gate.N_features,
                    N_qubits_tgt=gate.N_qubits_tgt)
                U_dagger = new_gate
            else:
                # Gate padrão
                try:
                    new_gate = gate.__class__(*new_params)
                except TypeError:
                    new_gate = gate.__class__(
                        name=gate.name,
                        num_qubits=gate.num_qubits,
                        params=new_params
                    )
        
        extended_qc.append(new_gate, qargs, cargs)
    
    return extended_qc, U_dagger

def conj_reversed_qc_ail(qc: QuantumCircuit):
    rev_ops = reversed(qc.data)
    a=0
    U_dagger = None
    for gate, qargs, cargs in rev_ops:
        new_gate = gate
        if a==0:
            new_params = [Parameter(f'conj_{param.name}') for param in gate.params]
        if isinstance(gate, ParamInitializeGate):
            # For our custom gate, just append as-is (parameters will be bound later)
            new_gate = gate.__class__(num_qubits=gate.num_qubits,
                                params=new_params[:gate.N_features],
                                N_features=gate.N_features)
        elif hasattr(gate, 'N_features') and hasattr(gate, 'N_qubits_tgt'):
            # Caso especial para nosso gate personalizado
            new_gate = gate.__class__(name=gate.name,
                                    num_qubits=gate.num_qubits,
                                    params=new_params,
                                    N_features=gate.N_features,
                                    N_qubits_tgt=gate.N_qubits_tgt)
            U_dagger = new_gate
            # Original handling for other gates
            #new_gate = gate.inverse() if hasattr(gate, 'inverse') else gate
        a+=1
        qc.append(new_gate, qargs, cargs)
    return qc, U_dagger

class ParamInitializeGate(Gate):
    def __init__(self, num_qubits, params, N_features):
        super().__init__("param_init", num_qubits, params)
        self.N_features = N_features
        
    def _define(self):
        q = QuantumRegister(self.num_qubits)
        qc = QuantumCircuit(q)
        
        # Convert parameters to normalized state vector
        params = np.array(self.params, dtype=complex)
        norm = np.linalg.norm(params)
        if norm > 0:
            params = params/norm
            
        qc.initialize(params, q[:])
        self.definition = qc

def circuitm(model: str, N_features, N_qubits, N_qubits_tgt, params, N_layers=None):
    if model == 'IQC':
        qc = QuantumCircuit(N_qubits, N_qubits_tgt)
        qc.h(range(N_qubits))
        
        class IQC_UGate(Gate):
            def __init__(self, name, num_qubits, params, N_features, N_qubits_tgt):
                super().__init__(name, num_qubits, params)
                self.N_features = N_features
                self.N_qubits_tgt = N_qubits_tgt
                
            def _define(self):
                q = QuantumRegister(self.num_qubits, 'q')
                qc = QuantumCircuit(q)
                param_values = [0]*len(self.params)  # Valores temporários
                U = get_U_operator_altered(param_values, self.N_features, self.num_qubits, self.N_qubits_tgt)
                qc.unitary(U, range(self.num_qubits))
                self.definition = qc
            def validate_parameter(self, parameter):
                return parameter  # Aceita qualquer parâmetro
        
        unitary_gate = IQC_UGate(f'U_{model}', N_qubits, params, N_features, N_qubits_tgt)
        qc.append(unitary_gate, range(N_qubits))

    if model == 'IQCpQ':
        qc = QuantumCircuit(N_qubits, N_qubits_tgt)
        qc.h(range(N_qubits))
        
        class IQCpQ_UGate(Gate):
            def __init__(self, name, num_qubits, params, N_features, N_qubits_tgt):
                super().__init__(name, num_qubits, params)
                self.N_features = N_features
                self.N_qubits_tgt = N_qubits_tgt
                
            def _define(self):
                q = QuantumRegister(self.num_qubits, 'q')
                qc = QuantumCircuit(q)
                param_values = [0]*len(self.params)  # Valores temporários
                U = get_U_operator_altered(param_values, self.N_features, self.num_qubits, self.N_qubits_tgt)
                qc.unitary(U, range(self.num_qubits))
                self.definition = qc
            def validate_parameter(self, parameter):
                return parameter  # Aceita qualquer parâmetro
        
        unitary_gate = IQCpQ_UGate(f'U_{model}', N_qubits, params, N_features, N_qubits_tgt)
        qc.append(unitary_gate, range(N_qubits))
    
    if model == 'IQCNDsE':
        qc = QuantumCircuit(N_qubits, N_qubits_tgt)
        qc.h(range(N_qubits))
        
        class IQCNDsE_UGate(Gate):
            def __init__(self, name, num_qubits, params, N_features, N_qubits_tgt):
                super().__init__(name, num_qubits, params)
                self.N_features = N_features
                self.N_qubits_tgt = N_qubits_tgt
                
            def _define(self):
                q = QuantumRegister(self.num_qubits, 'q')
                qc = QuantumCircuit(q)
                param_values = [0]*len(self.params)  # Valores temporários
                U = get_U_operator_altered(param_values, self.N_features, self.num_qubits, self.N_qubits_tgt, iqcndse=True)
                qc.unitary(U, range(self.num_qubits))
                self.definition = qc
            def validate_parameter(self, parameter):
                return parameter  # Aceita qualquer parâmetro
        
        unitary_gate = IQCNDsE_UGate(f'U_{model}', N_qubits, params, N_features, N_qubits_tgt)
        qc.append(unitary_gate, range(N_qubits))

    if model == 'IQC_AIL':
        qc = QuantumCircuit(N_qubits,N_qubits_tgt)
        init_gate = ParamInitializeGate(N_qubits-1, params[:N_features], N_features=N_features)
        qc.append(init_gate, range(1,N_qubits))
        qc.h(0)
        
        class IQC_AIL_UGate(Gate):
            def __init__(self, name, num_qubits, params, N_features, N_qubits_tgt):
                super().__init__(name, num_qubits, params)
                self.N_features = N_features
                self.N_qubits_tgt = N_qubits_tgt
                
            def _define(self):
                q = QuantumRegister(self.num_qubits, 'q')
                qc = QuantumCircuit(q)
                param_values = [0]*len(self.params)  # Valores temporários
                U = get_U_operator_altered(param_values, self.N_features, self.num_qubits, self.N_qubits_tgt, iqcail=True)
                qc.unitary(U, range(self.num_qubits))
                self.definition = qc
            def validate_parameter(self, parameter):
                return parameter  # Aceita qualquer parâmetro
        
        unitary_gate = IQC_AIL_UGate(f'U_{model}', N_qubits, params, N_features, N_qubits_tgt)
        qc.append(unitary_gate, range(N_qubits))
    
    if model == 'IQC_Angle':
        qreg=QuantumRegister(N_qubits, 'q')
        creg=ClassicalRegister(N_qubits_tgt)
        qc = QuantumCircuit(qreg, creg)     

        # Reaplica Hadamard ao final
        qc.h(0)
        
        rx_params=params[:N_features]

        # Armazena sequência de CNOTs
        """Aplica RXs e CNOTs, armazenando sequência, com barreira ao final."""
        for l in range(N_layers):
            for idx, qubit in enumerate(range(1, N_qubits)):
                qc.rx(rx_params[idx], qreg[qubit])
                
            for i in range(1, N_qubits - 1):
                qc.cx(qreg[i], qreg[i + 1])
            
            # Adiciona barreira
            qc.barrier()  
          
    
        class IQC_Angle_UGate(Gate):
            def __init__(self, name, num_qubits, params, N_features, N_qubits_tgt):
                super().__init__(name, num_qubits, params)
                self.N_features = N_features
                self.N_qubits_tgt = N_qubits_tgt
                
            def _define(self):
                q = QuantumRegister(self.num_qubits, 'q')
                qc = QuantumCircuit(q)
                param_values = [0]*len(self.params)  # Valores temporários
                U = get_U_operator_altered(param_values, self.N_features, self.num_qubits, self.N_qubits_tgt, iqcangle=True)
                #U_sparse, U_dense = get_U_sparse(param_values[:N_features], param_values[N_features:])
                #qc.unitary(U_dense, range(self.num_qubits))
                qc.unitary(U, range(self.num_qubits))
                #qc = get_U_operator_trotterized(param_values, N_qubits=self.num_qubits, N_layers=self.N_layers)

                """self.definition = qc
                param_values = [param for param in self.params]
                N_features = self.N_features
                
                U = get_U_operator_altered(param_values, self.N_features, self.num_qubits, self.N_qubits_tgt, iqcangle=True)
                
                # Ensure the matrix has the correct dimensions
                expected_dim = 2**self.num_qubits
                
                if U.shape != (expected_dim, expected_dim):
                    print(f"Warning: Reshaping matrix from {U.shape} to ({expected_dim}, {expected_dim})")
                    
                    # Option 1: If matrix is too large, take the top-left block
                    if U.shape[0] >= expected_dim and U.shape[1] >= expected_dim:
                        U = U[:expected_dim, :expected_dim]
                    # Option 2: If matrix is too small, pad with identity
                    elif U.shape[0] < expected_dim or U.shape[1] < expected_dim:
                        U_padded = np.eye(expected_dim, dtype=complex)
                        U_padded[:U.shape[0], :U.shape[1]] = U
                        U = U_padded
                    else:
                        raise ValueError(f"Cannot reshape matrix {U.shape} to ({expected_dim}, {expected_dim})")
                
                qc = QuantumCircuit(self.num_qubits)
                qc.unitary(U, range(self.num_qubits))"""

                self.definition = qc
            def validate_parameter(self, parameter):
                return parameter  # Aceita qualquer parâmetro

    
        unitary_gate = IQC_Angle_UGate(f'U_{model}', N_qubits, params, N_features, N_qubits_tgt)
        qc.append(unitary_gate, range(N_qubits))

    if model=='IQC_AIL': 
        qc, U_dagger=conj_reversed_qc_ail(qc)
        return qc, unitary_gate, U_dagger
    elif model=='IQC_Angle':
        qc, U_dagger=conj_reversed_qc_angle(qc)
        return qc, unitary_gate, U_dagger
    else: 
        qc, U_dagger = conj_reversed_qc(qc)
        return qc

"""def haar_integral(num_qubits, simulation_samples, N_features=None, model=None):
    '''
    Return the calculation of Haar Integral for a specified number of simulation_samples.
    '''

    if model=='IQC_Angle':
        N_QUBITS=(N_features+1) #Nqubits do circuito
        N=2**N_QUBITS
    else:
        N = 2 ** num_qubits
    
    randunit_density = np.zeros((N, N), dtype=complex)

    zero_state = np.zeros(N, dtype=complex)
    zero_state[0] = 1

    for _ in range(simulation_samples):
        # Generate random unitary
        unitary = np.matrix(unitary_group.rvs(N))
        # Apply unitary to the zero state
        A = np.matmul(zero_state, unitary).reshape(-1, 1)
        # Accumulate density matrix
        randunit_density += np.kron(A, A.conj().T)

    # Normalize by number of samples
    randunit_density /= simulation_samples
    return randunit_density"""

"""# Função para calcular a integral do PQC
def pqc_integral_adapted(N_QUBITS, simulation_samples, counter, QUBITS, N_features, model=None, folder=None, N_qubits_tgt=None, N_layers=None):
    '''
    Calcula a integral de um PQC com parâmetros aleatórios.
    
    Args:
        N_QUBITS (int): Número de qubits no circuito.
        circuit_model (function): Função que gera o circuito com parâmetros ajustáveis.
        simulation_samples (int): Número de amostras para calcular a integral.

    Returns:
        np.ndarray: Matriz densidade aproximada pelo circuito.
    '''
    randunit_density = np.zeros((2 ** N_QUBITS, 2 ** N_QUBITS), dtype=complex)

    if model=='IQCpQ':

        for _ in range(simulation_samples):
            # Gere os parâmetros aleatórios
            tx = rng.random((1,N_features))  # Parâmetros para tx
            tw = rng2.random((1,N_features))  # Parâmetros para tw
            # Cria o circuito com os parâmetros fornecidos
            qc,_,_ = circuit_model(data=tx[0],contador=_,w=tw,counter=counter,qubits=QUBITS,N_qubits=N_QUBITS,N_features=N_features,model=model,folder=folder,N_qubits_tgt=N_qubits_tgt,N_layers=N_layers)

            # Simule o circuito para obter o vetor de estado
            statevector = Statevector.from_instruction(transpile(qc, Aer.get_backend("statevector_simulator")))

            # Reshape statevector para vetor coluna
            U = statevector.data.reshape(-1, 1)

            # Acumule a matriz densidade
            randunit_density += np.kron(U, U.conj().T)

        # Normalize pela quantidade de amostras
        return randunit_density / simulation_samples
    elif model=='IQC_AIL':

        for _ in range(simulation_samples):
            # Gere os parâmetros aleatórios
            tx = rng.random((1,N_features))  # Parâmetros para tx
            tw = rng2.random((1,N_features))  # Parâmetros para tw
            tx=normalize_model(tx,model=model,normalize_col=False,normalize_lin=True)

            # Cria o circuito com os parâmetros fornecidos
            qc,_,_ = circuit_model(data=tx[0],contador=_,w=tw,counter=counter,qubits=QUBITS,N_qubits=N_QUBITS,N_features=len(tx[0]),model=model,folder=folder,N_qubits_tgt=N_qubits_tgt,N_layers=N_layers)

            # Simule o circuito para obter o vetor de estado
            statevector = Statevector.from_instruction(transpile(qc, Aer.get_backend("statevector_simulator")))

            # Reshape statevector para vetor coluna
            U = statevector.data.reshape(-1, 1)

            # Acumule a matriz densidade
            randunit_density += np.kron(U, U.conj().T)

        # Normalize pela quantidade de amostras
        return randunit_density / simulation_samples
    elif model=='IQC_Angle':
        N_QUBITS=(N_features+1) #Nqubits do circuito
        QUBITS=[i for i in range(N_QUBITS)]
        randunit_density = np.zeros((2 ** N_QUBITS, 2 ** N_QUBITS), dtype=complex)
        for _ in range(simulation_samples):
            # Gere os parâmetros aleatórios
            tx = rng.random((1,N_features))  # Parâmetros para tx
            tw = rng2.random((1,N_features))  # Parâmetros para tw

            # Cria o circuito com os parâmetros fornecidos
            qc,_,_ = circuit_model(data=tx[0],contador=_,w=tw,counter=counter,qubits=QUBITS,N_qubits=N_QUBITS,N_features=N_features,model=model,folder=folder,N_qubits_tgt=N_qubits_tgt,N_layers=N_layers)

            # Simule o circuito para obter o vetor de estado
            statevector = Statevector.from_instruction(transpile(qc, Aer.get_backend("statevector_simulator")))

            # Reshape statevector para vetor coluna
            U = statevector.data.reshape(-1, 1)

            # Acumule a matriz densidade
            randunit_density += np.kron(U, U.conj().T)

        # Normalize pela quantidade de amostras
        return randunit_density / simulation_samples
    else:
        for _ in range(simulation_samples):
            # Gere os parâmetros aleatórios
            tx = rng.random((1,N_features))  # Parâmetros para tx
            tw = rng2.random((1,N_features))  # Parâmetros para tw

            # Cria o circuito com os parâmetros fornecidos
            qc,_,_ = circuit_model(data=tx[0],contador=_,w=tw,counter=counter,qubits=QUBITS,N_qubits=N_QUBITS,N_features=N_features,model=model,folder=folder,N_qubits_tgt=N_qubits_tgt,N_layers=N_layers)

            # Simule o circuito para obter o vetor de estado
            statevector = Statevector.from_instruction(transpile(qc, Aer.get_backend("statevector_simulator")))

            # Reshape statevector para vetor coluna
            U = statevector.data.reshape(-1, 1)

            # Acumule a matriz densidade
            randunit_density += np.kron(U, U.conj().T)

        # Normalize pela quantidade de amostras
        return randunit_density / simulation_samples
"""

"""def thrash:
elif model=='IQC_AIL_RU': # IQC_AIL with arbitrary U operator  

    # IQC
    X_new=np.array(data)
    if np.log2(N_features)%2!=0 and np.log2(N_features)!=1:
        for k in range(2**(N_qubits-1) - N_features):
            w=np.append(w,0)
            X_new=np.append(X_new,0)
        
    qc = QuantumCircuit(N_qubits)
    qc.initialize(X_new, range(1,N_qubits)) # Inicializaçao do estado inicial. Poderia ser qualquer estado.
    qc.h(0)

    # Random Unitary Operator
    if counter==0: #para gerar
        U = np.matrix(unitary_group.rvs(2**N_qubits)) # If N_features = 2**m, this does the sames as U=np.matrix(unitary_group.rvs(2*N_features))
        '''
        Here, we may obtain an operator that acts in more qubits than the necessary (eq. ). Thus, we would need to 
        enlarge features vector to obtain same dimensionality
        '''

    qc.unitary(U,qubits)
    if counter==0:
        qc.draw("mpl", filename=folder+f'/mpl_complete_U_NF{N_features}_{model}.svg')
    if printar_cirq==True:
        display(qc.draw('mpl')) #display(qc.draw("mpl", filename='./mpl_original.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos

    #qc.decompose().draw(output="mpl", style="clifford")
    tqc=transpile(qc, optimization_level=0, basis_gates=['u3', 'x', 'h', 'z', 'cx'],seed_transpiler=1)

    gate_val = 0
    u3_dir = {}
    for i, instruction in enumerate(tqc.data):
        if instruction.operation.name == 'u3':
            u3_dir['u3_'+str(gate_val)] = {'qubit':instruction.qubits[0], 'params': instruction.operation.params}
            gate_val +=1
            
    if printar_cirq and dict(tqc.count_ops())['u3']<=50:
        print(u3_dir)
        print()

    
    u3_params = []
    for i in range(len(u3_dir)):
        u3_params.append(u3_dir[f'u3_{i}']['params'])

    if dict(tqc.count_ops())['u3']<=50 and contador==0:
        tqc.draw("mpl", filename=folder+f'/mpl_transpiled{contador}_NF{N_features}_{model}.svg')

    if printar_cirq==True and dict(tqc.count_ops())['u3']<=50:
        print(dict(tqc.count_ops()))
        display(tqc.draw('mpl')) #displat(qc.draw('mpl', filename='./mpl_transpile.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos


    # Mostrando o vetor de estado 
    sv = Statevector(qc)
    rho=np.array(DensityMatrix(sv))
    rho_cog = partial_trace(sv, qubits[1:])
    if printar_cirq==True:
        print(rho_cog)

    
    return blochvector(rho_cog,matriz_pauli_x,matriz_pauli_y,matriz_pauli_z),u3_params, get_negativity(rho,[2, N_features])
"""