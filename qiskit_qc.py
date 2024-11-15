import os
import warnings

import numpy as np

import qiskit
from qiskit.circuit import QuantumCircuit,Parameter
from qiskit import transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, visualize_transition, plot_bloch_vector
from qiskit.circuit.library import UnitaryGate,Initialize
from qiskit.quantum_info import Statevector,partial_trace, DensityMatrix

from toqito import state_props

import qutip

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
labels=[r'$\theta$',r'$\phi$',r'$\lambda$']
colors = ['forestgreen','darkorange','dodgerblue','deeppink' ]

"""

Available models: 'IQC', 'IQC_AIL', 'IQCEsQ', 'IQCNDsE_wx', 'IQCNDsE_Dx', 'IQCNDsE_xw', 'IQC_AIL_U', and 'IQC_U_Dx'.

"""

def normalize_model(data, model=None, normalize_col=True, normalize_lin=False):
    if model==None:
        raise Exception("Input model is necessary. Available models: 'IQC', 'IQC_AIL', 'IQCEsQ', 'IQCNDsE_wx', 'IQCNDsE_Dx', 'IQCNDsE_xw', 'IQC_AIL_U', and 'IQC_U_Dx'.")
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
    
def blochvector(rho_cog,matriz_pauli_x,matriz_pauli_y,matriz_pauli_z):
    x_bloch = np.trace(matriz_pauli_x@rho_cog.data)
    y_bloch = np.trace(matriz_pauli_y@rho_cog.data)
    z_bloch = np.trace(matriz_pauli_z@rho_cog.data)
    return [x_bloch,y_bloch,z_bloch]
    
#Executar o circuito
def run_qasm_counts(qc, shots):
    qc.measure_all()
    qasm_simulator = Aer.get_backend("qasm_simulator")
    job = qasm_simulator.run(qc, shots=shots)
    result = job.result()
    return result.get_counts()

def get_negativity(rho, dim):
    """
        Returns the Negativity associated with densitiy matrix rho.
        See definition at: https://en.wikipedia.org/wiki/Negativity_(quantum_mechanics)
        See implementation at: https://toqito.readthedocs.io/en/latest/_autosummary/toqito.state_props.negativity.html
    """
    return state_props.negativity(rho, dim)

def circuit_model(data,contador,w,counter,qubits,N_qubits,N_features,model=None,folder=None,N_qubits_tgt=None,printar_cirq=False):
    
    data=normalize_model(data, model=model)

    if model=='IQC':
        X_new=np.array(data)
        if np.log2(N_features)%2!=0 and np.log2(N_features)!=1:
            for k in range(2**(N_qubits-1) - N_features):
                w=np.append(w,0)
                X_new=np.append(X_new,0)
            sigmaE=np.diag(X_new)*w.T
        else:
            sigmaE=np.diag(X_new)*w.T
        

        # IQC

        qc = QuantumCircuit(N_qubits)

        qc.h(0)
        qc.h(range(1,N_qubits))



        #Montando os sigmas

        matriz_pauli_x=np.array([[0,1],[1,0]]) # Matriz de Pauli x
        matriz_pauli_y=np.array([[0,-1j],[1j,0]]) # Matriz de Pauli y
        matriz_pauli_z=np.array([[1,0],[0,-1]]) # Matriz de Pauli z

        sigmaQ=matriz_pauli_x+matriz_pauli_y+matriz_pauli_z

        

        #Operador Unitário
        U=np.matrix(expMatrix(1j*np.kron(sigmaQ,sigmaE)))

        # qubitstarget = [i for i in range(Ntarget)] - > Desnecessário agora, mas interessante para fazer a generalização
        qc.unitary(U,qubits)
        if counter==0:
            qc.draw("mpl", filename=folder+f'/mpl_complete_U_NF{N_features}_{model}.svg')
        if printar_cirq==True:
            display(qc.draw('mpl')) #display(qc.draw("mpl", filename='./mpl_original.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos

        #qc.decompose().draw(output="mpl", style="clifford")
        tqc=transpile(qc, optimization_level=3, basis_gates=["u3", "cx"], seed_transpiler=1)

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
    
    elif model=='IQC_AIL':
        X_new=np.array(data)
        if np.log2(N_features)%2!=0 and np.log2(N_features)!=1:
            for k in range(2**(N_qubits-1) - N_features):
                w=np.append(w,0)
                X_new=np.append(X_new,0)
            sigmaE=np.diag(w)
        else:
            sigmaE=np.diag(w)
       

        # IQC_AIL

        qc = QuantumCircuit(N_qubits)
        qc.initialize(X_new, range(1,N_qubits))# Inicializaçao do estado inicial. Poderia ser qualquer estado.
        qc.h(0)


        #Montando os sigmas

        matriz_pauli_x=np.array([[0,1],[1,0]]) # Matriz de Pauli x
        matriz_pauli_y=np.array([[0,-1j],[1j,0]]) # Matriz de Pauli y
        matriz_pauli_z=np.array([[1,0],[0,-1]]) # Matriz de Pauli z

        sigmaQ=matriz_pauli_x+matriz_pauli_y+matriz_pauli_z

        

        #Operador Unitário
        U=np.matrix(expMatrix(1j*np.kron(sigmaQ,sigmaE)))

        # qubitstarget = [i for i in range(Ntarget)] - > Desnecessário agora, mas interessante para fazer a generalização
        qc.unitary(U,qubits)
        if counter==0:
            qc.draw("mpl", filename=folder+f'/mpl_complete_U_NF{N_features}_{model}.svg')
        if printar_cirq==True:
            display(qc.draw('mpl')) #display(qc.draw("mpl", filename='./mpl_original.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos

        #qc.decompose().draw(output="mpl", style="clifford")
        tqc=transpile(qc, optimization_level=3, basis_gates=["u3", "cx"], seed_transpiler=1)

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
        
    elif model=='IQCEsQ': # IQC Expanding sigmaQ
        N_qubits_env=N_qubits
        if N_qubits_tgt:
            N_qubits_env-=N_qubits_tgt

        X_new=np.array(data)
        if np.log2(N_features)%2!=0 and np.log2(N_features)!=1:
            for k in range(2**(N_qubits_env) - N_features):
                w=np.append(w,0)
                X_new=np.append(X_new,0)
            sigmaE=np.diag(X_new)*w.T
        else:
            sigmaE=np.diag(X_new)*w.T
        

        # IQC

        qc = QuantumCircuit(N_qubits_env+N_qubits_tgt)
        qc.h(range(0,N_qubits_env+N_qubits_tgt))

        
        #Montando os sigmas

        matriz_pauli_x=np.array([[0,1],[1,0]]) # Matriz de Pauli x
        matriz_pauli_y=np.array([[0,-1j],[1j,0]]) # Matriz de Pauli y
        matriz_pauli_z=np.array([[1,0],[0,-1]]) # Matriz de Pauli z

        sigmaQ=matriz_pauli_x+matriz_pauli_y+matriz_pauli_z

        

        #Operador Unitário
        U=np.matrix(expMatrix(1j*np.kron(sigmaQ,sigmaE)))

        # qubitstarget = [i for i in range(Ntarget)] - > Desnecessário agora, mas interessante para fazer a generalização
        qc.unitary(U,qubits)

        if counter==0:
            qc.draw("mpl", filename=folder+f'/mpl_complete_U_NF{N_features}_{model}.svg')
        if printar_cirq==True:
            display(qc.draw('mpl')) #display(qc.draw("mpl", filename='./mpl_original.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos

        #qc.decompose().draw(output="mpl", style="clifford")
        tqc=transpile(qc, optimization_level=3, basis_gates=["u3", "cx"], seed_transpiler=1)

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
        
    elif model=='IQCNDsE_wx': # IQC Non Diagonal sigmaE: sE=w.T@x
        
        X_new=np.array(data)
        if np.log2(N_features)%2!=0 and np.log2(N_features)!=1:
            for k in range(2**(N_qubits-1) - N_features):
                w=np.append(w,0)
                X_new=np.append(X_new,0)
            w=w.T
            sigmaE=w.T@X_new
        else:
            w=w.T
            sigmaE=w.T@X_new
        
        
        # IQC

        qc = QuantumCircuit(N_qubits)
        qc.h(range(0,N_qubits))



        #Montando os sigmas

        matriz_pauli_x=np.array([[0,1],[1,0]]) # Matriz de Pauli x
        matriz_pauli_y=np.array([[0,-1j],[1j,0]]) # Matriz de Pauli y
        matriz_pauli_z=np.array([[1,0],[0,-1]]) # Matriz de Pauli z

        sigmaQ=matriz_pauli_x+matriz_pauli_y+matriz_pauli_z

        

        #Operador Unitário
        U=np.matrix(expMatrix(1j*np.kron(sigmaQ,sigmaE)))

        # qubitstarget = [i for i in range(Ntarget)] - > Desnecessário agora, mas interessante para fazer a generalização
        qc.unitary(U,qubits)
        if counter==0:
            qc.draw("mpl", filename=folder+f'/mpl_complete_U_NF{N_features}_{model}.svg')
        if printar_cirq==True:
            display(qc.draw('mpl')) #display(qc.draw("mpl", filename='./mpl_original.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos

        #qc.decompose().draw(output="mpl", style="clifford")
        tqc=transpile(qc, optimization_level=3, basis_gates=["u3", "cx"], seed_transpiler=1)

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
        
    elif model=='IQCNDsE_Dx': # IQC Non Diagonal sigmaE: x elements occupy the diagonal of sigmaE

        X_new=np.array(data)
        if np.log2(N_features)%2!=0 and np.log2(N_features)!=1:
            for k in range(2**(N_qubits-1) - N_features):
                w=np.append(w,0)
                X_new=np.append(X_new,0)
         
        # Creating sigmaE: each row is the vector w
        sigmaE = np.tile(w, (N_features, 1))  # Repeat w for each row of the matrix
        # Modifying the diagonal of sigmaE by multiplying with X
        np.fill_diagonal(sigmaE, np.diag(sigmaE) * X_new)
        
        
        # IQC

        qc = QuantumCircuit(N_qubits)
        qc.h(range(0,N_qubits))

        #Montando os sigmas

        matriz_pauli_x=np.array([[0,1],[1,0]]) # Matriz de Pauli x
        matriz_pauli_y=np.array([[0,-1j],[1j,0]]) # Matriz de Pauli y
        matriz_pauli_z=np.array([[1,0],[0,-1]]) # Matriz de Pauli z

        sigmaQ=matriz_pauli_x+matriz_pauli_y+matriz_pauli_z

        

        #Operador Unitário
        U=np.matrix(expMatrix(1j*np.kron(sigmaQ,sigmaE)))

        # qubitstarget = [i for i in range(Ntarget)] - > Desnecessário agora, mas interessante para fazer a generalização
        qc.unitary(U,qubits)
        if counter==0:
            qc.draw("mpl", filename=folder+f'/mpl_complete_U_NF{N_features}_{model}.svg')
        if printar_cirq==True:
            display(qc.draw('mpl')) #display(qc.draw("mpl", filename='./mpl_original.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos

        #qc.decompose().draw(output="mpl", style="clifford")
        tqc=transpile(qc, optimization_level=3, basis_gates=["u3", "cx"], seed_transpiler=1)

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
    
    elif model=='IQCNDsE_xw': # IQC Non Diagonal sigmaE: sE=x.T@w  
        
        X_new=np.array(data)
        if np.log2(N_features)%2!=0 and np.log2(N_features)!=1:
            for k in range(2**(N_qubits-1) - N_features):
                w=np.append(w,0)
                X_new=np.append(X_new,0)
             
            sigmaE=X_new.T@w
        else:
             
            sigmaE=X_new.T@w
        

        # IQC

        qc = QuantumCircuit(N_qubits)
        qc.h(range(0,N_qubits))



        #Montando os sigmas

        matriz_pauli_x=np.array([[0,1],[1,0]]) # Matriz de Pauli x
        matriz_pauli_y=np.array([[0,-1j],[1j,0]]) # Matriz de Pauli y
        matriz_pauli_z=np.array([[1,0],[0,-1]]) # Matriz de Pauli z

        sigmaQ=matriz_pauli_x+matriz_pauli_y+matriz_pauli_z

        

        #Operador Unitário
        U=np.matrix(expMatrix(1j*np.kron(sigmaQ,sigmaE)))

        # qubitstarget = [i for i in range(Ntarget)] - > Desnecessário agora, mas interessante para fazer a generalização
        qc.unitary(U,qubits)
        if counter==0:
            qc.draw("mpl", filename=folder+f'/mpl_complete_U_NF{N_features}_{model}.svg')
        if printar_cirq==True:
            display(qc.draw('mpl')) #display(qc.draw("mpl", filename='./mpl_original.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos

        #qc.decompose().draw(output="mpl", style="clifford")
        tqc=transpile(qc, optimization_level=3, basis_gates=["u3", "cx"], seed_transpiler=1)

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
        
    elif model=='IQC_AIL_U': # IQC_AIL with arbitrary U operator  

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
        tqc=transpile(qc, optimization_level=3, basis_gates=["u3", "cx"], seed_transpiler=1)

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

    elif model=='IQC_U_Dx': # IQC with features vector embedded in U operator
        # IQC
        X_new=np.array(data)
        qc = QuantumCircuit(N_qubits)
        qc.h(range(0,N_qubits)) # State Initialization

        # Random Unitary Operator
        U = np.matrix(unitary_group.rvs(2**N_qubits))
        
        # Applying X to the diagonal of U
        U[:N_features, :N_features] *= np.diag(X_new)  # Multiply the top-left diagonal block
        U[N_features:, N_features:] *= np.diag(X_new)  # Multiply the bottom-right diagonal block
    
        qc.unitary(U,qubits)
        if counter==0:
            qc.draw("mpl", filename=folder+f'/mpl_complete_U_NF{N_features}_{model}.svg')
        if printar_cirq==True:
            display(qc.draw('mpl')) #display(qc.draw("mpl", filename='./mpl_original.pdf')) #Trocar as chamadas se quiser salvar as imagens dos circuitos

        #qc.decompose().draw(output="mpl", style="clifford")
        tqc=transpile(qc, optimization_level=3, basis_gates=["u3", "cx"], seed_transpiler=1)

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

    elif model==None:
        raise Exception("Input model is necessary. Available models: 'IQC', 'IQC_AIL', 'IQCEsQ', 'IQCNDsE_wx', 'IQCNDsE_Dx', 'IQCNDsE_xw', 'IQC_AIL_U', and 'IQC_U_Dx'.")
    
    if folder==None:
        raise Exception("No folder selected.")

# Define a function to check and crop possible lists with different sizes
def dividir_por_tamanho(lista):
    # Dicionário para armazenar sublistas agrupadas pelo tamanho
    grupos = {}
    
    for sublista in lista:
        tamanho = len(sublista)
        if tamanho not in grupos:
            grupos[tamanho] = []
        grupos[tamanho].append(sublista)
    
    return grupos

def esfera_bloch(X,weights,qubits,N_qubits,N_features,counter,model=None,folder=None,printar_esf=False,norma=None):
    if model==None:
        raise Exception("Input model is necessary. Available models: 'IQC', 'IQC_AIL', 'IQCEsQ', 'IQCNDsE_wx', 'IQCNDsE_Dx', 'IQCNDsE_xw', 'IQC_AIL_U', and 'IQC_U_Dx'.")
    
    if folder==None:
        raise Exception("No folder selected.")
    
    point_states=[]
    u3_params=[]
    negativity=[]
    for k in range(len(X)):
        bloch,params,neg=circuit_model(X[k],k,weights[k], counter, qubits, N_qubits, N_features,folder,model=model)
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

def plot_histogram(u3_list,neg_list,N_features,folder=None,norma=None, model=None):
    
    if model==None:
        raise Exception("Input model is necessary. Available models: 'IQC', 'IQC_AIL', 'IQCEsQ', 'IQCNDsE_wx', 'IQCNDsE_Dx', 'IQCNDsE_xw', 'IQC_AIL_U', and 'IQC_U_Dx'.")
    
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

    
    lista=dividir_por_tamanho(u3_list) # lista[G size index][tqc index][gate index][params] --> lista[int(np.unique[j])][:][i][0 or 1 or 2]
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
    elif model=='IQCEsQ': # IQC Expanding sigmaQ
        model_title = 'IQCEsQ'
    elif model=='IQCNDsE_wx': # IQC Non Diagonal sigmaE: sE=w.T@x
        model_title = 'IQCNDsE_wx'
    elif model=='IQCNDsE_Dx': # IQC Non Diagonal sigmaE: x elements occupy the diagonal of sigmaE
        model_title = 'IQCNDsE_Dx'
    elif model=='IQCNDsE_xw': # IQC Non Diagonal sigmaE: sE=x.T@w  
        model_title = 'IQCNDsE_xw'    
    elif model=='IQC_AIL_U': # IQC_AIL with arbitrary U operator
        model_title = 'IQC_AIL_U'
    elif model=='IQC_U_Dx': # IQC with features vector embedded in U operator
        model_title = 'IQC_U_Dx'
    elif model==None:
        raise Exception("Input model is necessary. Available models: 'IQC', 'IQC_AIL', 'IQCEsQ', 'IQCNDsE_wx', 'IQCNDsE_Dx', 'IQCNDsE_xw', 'IQC_AIL_U', and 'IQC_U_Dx'.")

    num_neg=[]
    for i in range(N_samples):
        num_neg.append(i)

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

        ax[0].scatter(num_neg,neg_list1, marker='.', s=12)
        ax[0].set_ylabel('Negativity Value')
        plt.savefig(folder+f'/Negativity_NF{N_features}_{model}.svg')
        plt.close(fig)

def statistical_qc(N_samples,N_features,model=None,folder=None,normalization=False):
    if folder==None:
        raise Exception("No folder selected.")
    if model==None:
        raise Exception("Input model is necessary. Available models: 'IQC', 'IQC_AIL', 'IQCEsQ', 'IQCNDsE_wx', 'IQCNDsE_Dx', 'IQCNDsE_xw', 'IQC_AIL_U', and 'IQC_U_Dx'.")
    
    counter=0
    X_df=np.random.rand(N_samples,N_features)
    w_df=np.random.rand(N_samples,N_features)
    N_qubits=math.ceil(np.log2(N_features)+1) #Nqubits do circuito
    qubits=[i for i in range(N_qubits)]

    if normalization:
        X_df_coluna=normalize_model(X_df, normalize_col=True, normalize_lin=False)
        X_df_linha=normalize_model(X_df,normalize_col=False,normalize_lin=True)
        u3_col,neg_col=esfera_bloch(X_df_coluna,w_df,qubits,N_qubits,N_features,counter,folder,norma='coluna',model=model)
        u3_lin,neg_lin=esfera_bloch(X_df_linha,w_df,qubits,N_qubits,N_features,counter,folder,norma='linha',model=model)
        return u3_col, u3_lin, neg_col, neg_lin
    else:
        X_df_iqc_ail=normalize_model(X_df,normalize_col=False,normalize_lin=True)
        u3_lista,neg_lista=esfera_bloch(X_df_iqc_ail,w_df,qubits,N_qubits,N_features,counter,folder,model=model)
        return u3_lista, neg_lista
    