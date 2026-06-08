import numpy as np

import qiskit
from qiskit.circuit import QuantumCircuit,Parameter, QuantumRegister, ClassicalRegister, Gate, Measure, ParameterVector
from qiskit import transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, visualize_transition, plot_bloch_vector
from qiskit.circuit.library import UnitaryGate,Initialize
from qiskit.quantum_info import Statevector,partial_trace, DensityMatrix, Operator

from scipy.linalg import expm as expMatrix
import scipy.linalg
from sympy.physics.quantum.dagger import Dagger
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

import matplotlib.pyplot as plt

from scipy.special import rel_entr
from scipy.stats import entropy
from scipy.special import kl_div, rel_entr

plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout': True})
colors = ['forestgreen','darkorange','dodgerblue','deeppink' ]

rng=np.random.default_rng(1)
rng2=np.random.default_rng(42)

def available_qc():
    print("Available models: 'IQC', 'IQC_AIL', 'IQCETS', 'IQC_Angle', and 'IQCNDsE'.")

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

# Execute qiskit circuit measuring only the target qubits
def run_qasm_counts(qc, shots, N_qubits_tgt, backend='qasm_simulator'):
    qc.measure([i for i in range(N_qubits_tgt)],[i for i in range(N_qubits_tgt)])
    qasm_simulator = Aer.get_backend(backend)
    job = qasm_simulator.run(qc, shots=shots)
    result = job.result()
    return result.get_counts()

# Execute qiskit circuit measuring all qubits
def run_qasm_counts_meas_all(qc, shots):
    qc.measure_all()
    qasm_simulator = Aer.get_backend("qasm_simulator")
    job = qasm_simulator.run(qc, shots=shots)
    result = job.result()
    return result.get_counts()

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

def get_U_operator_altered(params, 
                           N_features, 
                           N_qubits, 
                           N_qubits_tgt, 
                           iqcail=False, 
                           iqcndse=False, 
                           iqcangle=False):
    
    X = params[:N_features]
    vw = params[N_features:]
    
    # Calcular o tamanho esperado de forma consistente
    expected_size = 2 ** (N_qubits - N_qubits_tgt)
    
    # Converter para arrays numpy
    X_new = np.array(X, dtype=float)
    w = np.array(vw, dtype=float)
    
    # Truncar ou expandir para o tamanho esperado (NÃO duplicar padding)
    if len(X_new) != expected_size:
        if len(X_new) > expected_size:
            X_new = X_new[:expected_size]
            w = w[:expected_size]
        else:
            # Se for menor, faz padding
            pad_size = expected_size - len(X_new)
            X_new = np.pad(X_new, (0, pad_size), constant_values=0)
            w = np.pad(w, (0, pad_size), constant_values=0)
    
    # Garantir que ambos têm o mesmo comprimento
    assert len(X_new) == len(w) == expected_size, f"Size mismatch: X={len(X_new)}, w={len(w)}, expected={expected_size}"
    
    if iqcail:
        sigmaE = np.diag(w)
        
    elif iqcndse:
        X_new = np.matrix(X_new)
        w = np.matrix(w)
        sigmaE = X_new.T @ w + (X_new.T @ w).T
        
    elif iqcangle:
        sigmaE = np.diag(w)
        dim_circuit = 2 ** (N_qubits - 1)
        dim_sigmaE = sigmaE.shape[0]
        sigmaE = np.kron(np.eye(dim_circuit // dim_sigmaE), sigmaE)
        
    else:
        # Criar sigmaE com a dimensionalidade correta
        # sigmaE deve ser uma matriz diagonal onde cada elemento diagonal é X_new[i] * w[i]
        sigmaE = np.diag(X_new * w)  # Multiplicação elemento a elemento
    
    # Construção do sigmaQ
    if N_qubits_tgt == 1:
        sigma_q_params = np.full(2 ** N_qubits_tgt, 1)
        sigmaQ = get_weighted_sigmaQ(sigma_q_params, iqcpq=False)
    else:
        sigma_q_params = np.full(2 ** N_qubits_tgt, 1)
        sigmaQ = get_weighted_sigmaQ(sigma_q_params, iqcpq=True)
    
    # Operador Unitário
    kron_product = np.kron(sigmaQ, sigmaE)
    U = np.matrix(expMatrix(1j * kron_product))
    
    return U

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

class ParamInitializeGateOLD(Gate):
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

class ParamInitializeGate(Gate):
    def __init__(self, num_qubits, params, N_features):
        super().__init__("param_init", num_qubits, params)
        self.N_features = N_features
        
    def _define(self):
        q = QuantumRegister(self.num_qubits)
        qc = QuantumCircuit(q)
        
        # Convert parameters to normalized state vector
        params_array = np.array([float(p) for p in self.params], dtype=complex)
        norm = np.linalg.norm(params_array)
        if norm > 0:
            params_array = params_array / norm
        
        # Verificar se o comprimento é uma potência de 2
        n_qubits_needed = int(np.ceil(np.log2(len(params_array))))
        if len(params_array) != 2**n_qubits_needed:
            # Fazer padding para a próxima potência de 2
            target_size = 2**n_qubits_needed
            padded = np.zeros(target_size, dtype=complex)
            padded[:len(params_array)] = params_array
            params_array = padded
            # Ajustar o número de qubits
            qc = QuantumCircuit(n_qubits_needed)
            qc.initialize(params_array, range(n_qubits_needed))
        else:
            qc.initialize(params_array, q[:])
        
        self.definition = qc
    
    def __str__(self):
        return f"param_init({self.params})"
    
    # Método para evitar decomposição automática
    def inverse(self):
        return self
    
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

    elif model == 'IQCpQ':
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
    
    elif model == 'IQCNDsE':
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

    
    elif model == 'IQC_AIL':
        # Criar o circuito
        qc = QuantumCircuit(N_qubits, N_qubits_tgt)
        
        # Verificar o tamanho necessário para a inicialização
        n_env_qubits = N_qubits - 1
        required_size = 2 ** n_env_qubits
        
        # Ajustar os parâmetros para o tamanho correto
        init_params = params[:N_features]
        
        # Se o número de features não corresponde ao espaço de Hilbert, fazer padding
        if N_features < required_size:
            # Adicionar parâmetros zero para padding
            padding_params = [Parameter(f'pad_{i}') for i in range(required_size - N_features)]
            all_init_params = list(init_params) + padding_params
        else:
            all_init_params = list(init_params)[:required_size]
        
        # Criar o gate de inicialização
        init_gate = ParamInitializeGate(n_env_qubits, all_init_params, N_features=len(all_init_params))
        qc.append(init_gate, range(1, N_qubits))
        qc.h(0)
        
        # Gate U personalizado para IQC_AIL
        class IQC_AIL_UGate(Gate):
            def __init__(self, name, num_qubits, params, N_features, N_qubits_tgt):
                super().__init__(name, num_qubits, params)
                self.N_features = N_features
                self.N_qubits_tgt = N_qubits_tgt
                
            def _define(self):
                q = QuantumRegister(self.num_qubits, 'q')
                qc_def = QuantumCircuit(q)
                param_values = [0] * len(self.params)
                U = get_U_operator_altered(param_values, self.N_features, self.num_qubits, self.N_qubits_tgt, iqcail=True)
                qc_def.unitary(U, range(self.num_qubits))
                self.definition = qc_def
                
            def validate_parameter(self, parameter):
                return parameter
            
            def __str__(self):
                return f"U_IQC_AIL"
            
            def inverse(self):
                return self
        
        unitary_gate = IQC_AIL_UGate(f'U_{model}', N_qubits, params, N_features, N_qubits_tgt)
        qc.append(unitary_gate, range(N_qubits))
        
        # Adicionar reverso conjugado (se necessário)
        # qc, _, _ = conj_reversed_qc_ail(qc)  # Comente esta linha se causar problemas
        
        return qc
    
    elif model == 'IQC_Angle':
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
        return qc#, unitary_gate, U_dagger
    elif model=='IQC_Angle':
        qc, U_dagger=conj_reversed_qc_angle(qc)
        return qc, unitary_gate, U_dagger
    else: 
        qc, U_dagger = conj_reversed_qc(qc)
        return qc
    
def expressibility(MODEL, NF, print_hist=False, print_circuit=False, n_shots=8192, simulation_samples=1000):
    if MODEL == 'IQC_Angle':
        N_qubits_tgt = 1
        N_qubits = (NF + N_qubits_tgt)
    elif MODEL == 'IQCpQ':
        N_qubits_tgt = 2
        N_qubits = math.ceil(np.log2(NF) + N_qubits_tgt)
    else:
        N_qubits_tgt = 1
        N_qubits = math.ceil(np.log2(NF) + N_qubits_tgt)
    
    string_zero = '0' * N_qubits  # String de controle para calcular a fidelidade do circuito
    
    P_harr_hist, bins_x, bins_list = bins(N_qubits=N_qubits)
    
    folder = f'NF{NF}_{MODEL}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # CRIAÇÃO DOS PARÂMETROS SEM PADDING AQUI
    # O padding será feito dentro de get_U_operator_altered
    tx = [Parameter(f'x{i}') for i in range(NF)]
    tw = [Parameter(f'pw{i}') for i in range(NF)]
    
    params = tx + tw
    
    if MODEL == 'IQC_Angle':
        qc = circuitm(MODEL, NF, N_qubits, N_qubits_tgt, params=params, N_layers=2)
    else:
        qc = circuitm(MODEL, NF, N_qubits, N_qubits_tgt, params=params)
    
    fidelity = []    
    for _ in range(simulation_samples):
        if MODEL == 'IQC_Angle':
            param_binding = {p: np.pi * rng.random() for p in qc.parameters}
        else:
            param_binding = {p: rng.random() for p in qc.parameters}
        
        bound_qc = qc.assign_parameters(param_binding)
        
        # CORREÇÃO: Tentar decompor com tratamento de erro mais robusto
        REPS=100
        try:
            decomposed = bound_qc.decompose()  # Começar com reps=1
            # Se funcionar, tentar mais decomposições
            for _ in range(REPS-1):  # reps é o número desejado
                decomposed = decomposed.decompose()
        except Exception as e:
            print(f"Warning: decompose failed ({e}), using bound circuit directly")
            decomposed = bound_qc
        
        tqc = transpile(bound_qc, optimization_level=0, 
                       basis_gates=['u3', 'x', 'h', 'z', 'cx'], 
                       seed_transpiler=1)
        
        count = run_qasm_counts(tqc, n_shots, N_qubits_tgt)
        
        if string_zero in count:
            ratio = count[string_zero] / n_shots
        else:
            ratio = 0
        fidelity.append(ratio)
    
    if print_circuit:
        qc.draw('mpl', filename = folder + f'/mpl_meas_tgt_U_NF{NF}_{MODEL}.svg')
        plt.close()
    
    weights = np.ones_like(fidelity) / float(len(fidelity))
    
    if print_hist:
        plt.hist(fidelity, bins=bins_list, weights=weights, range=[0, 1], label=MODEL)
        plt.plot(bins_x, P_harr_hist, label='Haar')
        plt.legend(loc='upper right')
        plt.ylabel('Probability')
        plt.xlabel('Fidelity')
        plt.title(f'Fidelity Distribution - NF{NF} - {MODEL}')
        plt.tight_layout()
        plt.savefig(folder + f'/Expressibility_Hist_NF{NF}_{MODEL}.svg')
        plt.show()
    
    # Calcular histograma e KL divergence
    P_I_hist, _ = np.histogram(fidelity, bins=bins_list, weights=weights, range=[0, 1])
    # Adicionar pequeno epsilon para evitar log(0)
    P_harr_hist = np.array(P_harr_hist)
    P_I_hist = P_I_hist
    # Normalizar
    P_harr_hist = P_harr_hist / np.sum(P_harr_hist)
    P_I_hist = P_I_hist / np.sum(P_I_hist)
    
    kl_pq = entropy(P_I_hist, P_harr_hist)
    print('KL Divergence (P || Q) = %.5f' % kl_pq)
    
    return kl_pq
