import sympy as sp
import numpy as np
from array import array
from itertools import product
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt



def forget(T, pos):
    ### This method takes in input an n-order tensor T and gives as output a n+1-order tensor T
    ### pos is an integer in the interval [0,len(T.shape)] 
    if pos < 0 or pos > len(T.shape):
        raise ValueError("Invalid value for pos")
    new_format = array("i",[2 for _ in range(len(T.shape)+1)])
    T_out = sp.MutableDenseNDimArray(np.zeros(new_format))
    indices = product([0,1], repeat=len(T.shape))
    for index in indices:
        
        ind1= list(index)[:pos]
        ind2 = list(index)[pos:]
        T_out[ind1+[0]+ind2] = T[index]
        T_out[ind1+[1]+ind2] = T[index]
    return T_out
    

def blow(T):
    new_format = array("i",[2 for _ in range(len(T.shape)+1)])
    T_out = sp.MutableDenseNDimArray(np.zeros(new_format))
    indices = product([0,1], repeat=len(T.shape))
    for index in indices:
        for i in range(2):
            if i == index[0]:
                T_out[list(index)+[i]] = T[index]
            else:
                T_out[list(index)+[i]] = 0
    return T_out

def BM_product(tensor_list):
    # tensor_list must be ordered according to the order requested for the BM product
    # up to now it takes only tensors with indices in {0,1}
    tensor_list = [tensor_list.pop()]+tensor_list
    indices = product([0,1], repeat=len(tensor_list[0].shape))
    T_out = sp.MutableDenseNDimArray(np.zeros(tensor_list[0].shape))
    for index in indices:
        for i in range(2):
            add = 1
            for j in range(len(tensor_list[0].shape)):
                index_copy = list(index)
                index_copy[j] = i
                add = add*tensor_list[j][index_copy]
            T_out[index] += add
    return T_out

def general_forget(T, pos):
    N = T.shape[0] # assuming T is cubical 
    ### This method takes in input an n-order tensor T and gives as output a n+1-order tensor T
    ### pos is an integer in the interval [0,len(T.shape)] 
    if pos < 0 or pos > len(T.shape):
        raise ValueError("Invalid value for pos")
    new_format = array("i",[N for _ in range(len(T.shape)+1)])
    T_out = sp.MutableDenseNDimArray(np.zeros(new_format))
    indices = product([i for i in range(N)], repeat=len(T.shape))
    for index in indices:      
        ind1= list(index)[:pos]
        ind2 = list(index)[pos:]
        for i in range(N):
            T_out[ind1+[i]+ind2] = T[index]

    return T_out
    

def general_blow(T):
    N = T.shape[0] # assuming T is cubical
    new_format = array("i",[N for _ in range(len(T.shape)+1)])
    T_out = sp.MutableDenseNDimArray(np.zeros(new_format))
    indices = product([i for i in range(N)], repeat=len(T.shape))
    for index in indices:
        for i in range(N):
            if i == index[0]:
                T_out[list(index)+[i]] = T[index]
            else:
                T_out[list(index)+[i]] = 0
    return T_out


def general_BM_product(tensor_list):
    N = tensor_list[0].shape[0]
    tensor_list = [tensor_list.pop()]+tensor_list
    # tensor_list must be ordered according to the order requested for the BM product
    # valid for tensors with indices in {0,...,N-1}
    indices = product([i for i in range(N)], repeat=len(tensor_list[0].shape))
    T_out = sp.MutableDenseNDimArray(np.zeros(tensor_list[0].shape))
    for index in indices:
        for i in range(N):
            add = 1
            for j in range(len(tensor_list[0].shape)):
                index_copy = list(index)
                index_copy[j] = i
                add = add*tensor_list[j][index_copy]
            T_out[index] += add
    return np.array(T_out)

def get_tensor_operations_indexed(i, T, adjacency_matrix):
    """
    Automates the identification of forgets (J, H) and blows
    using 0-based indexing (0 to q-1).
    
    Args:
        i: Current node index (0 to q-1)
        q: Total number of nodes
        adjacency_matrix: q x q matrix (A[j][i] == 1 if j is parent of i)
    """
    # Step 1: Identify Parents (Pi)
    # Check column i for rows j < i
    q = adjacency_matrix.shape[0] # Assuming A is the adjacency matrix
    pi = [j for j in range(i) if adjacency_matrix[j][i] == 1]
    
    operations = []
    
    # Step 2: Forget Previous (J)
    # Target dimension after this step: i + 1
    if i > 0:
        pre_nodes = set(range(i))
        j_set = sorted(list(pre_nodes - set(pi)))
        for index in j_set:
            T = forget(T, index)
        operations.append(f"Step 1 (Forget J): {j_set} -> Tensor order after forget: {T.shape}")

    # Step 3: Blow Link (b)
    # Target dimension after this step: i + 2
    if i < q - 1:
        T = blow(T)
        operations.append(f"Step 2 (Blow): Inflate to order {i+2} (Link index 0 == index {i+1})")

    # Step 4: Forget Future (H)
    # Target dimension after this step: q
    if i + 1 < q - 1:
        h_set = list(range(i + 2, q))
        for index in h_set:
            T = forget(T, index)
        operations.append(f"Step 3 (Forget H): {h_set} -> Final tensor order {q}")

    return {
        "node_index": i,
        "parent_indices": pi, 
        "tensor": T,
        "op_sequence": operations
    }

def edge_index_to_adj(edge_index, q):
    """
    Converts an edge_index (2 x E) to a q x q Adjacency Matrix.
    
    Args:
        edge_index: List or array of shape (2, E) 
                    edge_index[0] = source nodes
                    edge_index[1] = target nodes
        q: The total number of nodes in the graph
        
    Returns:
        adj: A q x q numpy array where adj[j][i] = 1 if there's an edge j -> i
    """
    # Initialize a q x q matrix of zeros
    adj = np.zeros((q, q), dtype=int)
    
    # Extract sources and targets
    sources = edge_index[:,0]
    targets = edge_index[:,1]
    
    # Fill the matrix: A[source][target] = 1
    # Since your definitions use A[j][i] for j being a parent of i
    for j, i in zip(sources, targets):
        if j < q and i < q:
            adj[j][i] = 1
            
    return adj

def create_random_tensors(adjacency_matrix, params = [sp.symbols("alpha"), sp.symbols("beta")]):
    '''
    Creates a list of tensors according to the structure of the adjacency matrix.
    Each tensor is initialized with the same parameters and has an order equal to the number of parents of the corresponding node + 1.
    '''
    q = adjacency_matrix.shape[0]
    T_list = []
    for j in range(q):
        T = sp.MutableDenseNDimArray(params)
        for _ in range(adjacency_matrix[:,j].sum()):
            T = [T, T[::-1]]
        T_list.append(np.array(T))
    return T_list

def create_network(adjacency_matrix, tensors_list):
    '''
    Creates a list of tensors according to the structure of the adjacency matrix and the input tensors list.
    Each tensor is transformed according to the operations defined by the adjacency matrix and the position of the node in the graph.
    '''
    q = adjacency_matrix.shape[0]
    network = []
    for i in range(q):
        network.append(get_tensor_operations_indexed(i, tensors_list[i], adjacency_matrix)['tensor'])
    return network

def flat(T):

    length = len(T.shape)
    if length<3:
        raise Exception('tensor is not 3D')
    new_shape = [8]+[2 for _ in range(length-3)]
    new_T = sp.MutableDenseNDimArray(np.zeros(new_shape))
    if length-3>0:
        indices = product([0,1], repeat=length-3)
    else:
        indices = [[]]
    for index in indices:
        new_T[[0]+list(index)] = T[[0,0,0]+list(index)]
        new_T[[1]+list(index)] = T[[1,0,0]+list(index)]
        new_T[[2]+list(index)] = T[[0,1,0]+list(index)]
        new_T[[2]+list(index)] = T[[1,1,0]+list(index)]
        new_T[[4]+list(index)] = T[[0,0,1]+list(index)]
        new_T[[5]+list(index)] = T[[1,0,1]+list(index)]
        new_T[[6]+list(index)] = T[[0,1,1]+list(index)]
        new_T[[7]+list(index)] = T[[1,1,1]+list(index)]
    return new_T

def slice_index(T, index):
    # return T[:,:,:,[index]]
    new_T = sp.MutableDenseNDimArray(np.zeros([8]))
    new_T[[0]] = T[[0,0,0]+list(index)]
    new_T[[1]] = T[[1,0,0]+list(index)]
    new_T[[2]] = T[[0,1,0]+list(index)]
    new_T[[3]] = T[[1,1,0]+list(index)]
    new_T[[4]] = T[[0,0,1]+list(index)]
    new_T[[5]] = T[[1,0,1]+list(index)]
    new_T[[6]] = T[[0,1,1]+list(index)]
    new_T[[7]] = T[[1,1,1]+list(index)]
    return new_T
    # 

def draw_tensor(T,ax):
    # draw tensors of order 3 with indices in {0,1}
    ax.plot([5.6, 9.4],[2,2],'k')
    ax.plot([7.6,11.4],[4,4],'k')
    ax.plot([5.3,6.7],[2.3,3.7],'k')
    ax.plot([10.6,11.7],[2.6,3.7],'k')
    ax.plot([5,5],[2.6,6.4],'k')
    ax.plot([7,7],[4.6,8.4],'k')
    ax.plot([10,10],[2.6,6.4],'k')
    ax.plot([12,12],[4.6,8.4],'k')
    ax.plot([5.6,9.4],[7,7],'k')
    ax.plot([10.6,11.7],[7.6,8.7],'k')
    ax.plot([5.6,6.7],[7.6,8.7],'k')
    ax.plot([7.6,11.4],[9,9],'k')
    x=[5, 5, 10, 10, 7, 7, 12, 12]
    y=[7, 2, 7, 2, 9, 4, 9, 4]
    for i in range(len(x)):
        ax.text(x[i], y[i], T[i])

def draw_general(T,name):
    #  draws tensors up to order 5
    length = len(T.shape)
    fig, axs = plt.subplots(2**(int(np.floor((length-3)/2))),2**(int(np.floor((length-2)/2))))
    #print(axs)
    if length-3>0:
        for i,ax in enumerate(axs.flat):
            index = np.unravel_index(i, [2 for _ in range(length-3)])
            print(index)
            ax.set_aspect('equal')
            ax.axis('off')
            draw_tensor(slice_index(T,list(index)),ax)
    else:
        axs.set_aspect('equal')
        axs.axis('off')
        index = []
        draw_tensor(slice_index(T,list(index)),axs)
            

    fig.tight_layout()
    plt.savefig(name+'.pdf')


if __name__== '__main__':
    from utils import *
    import sympy as sp
    a = sp.symbols("a")
    b = sp.symbols("b")
    c = sp.symbols("c")
    d = sp.symbols("d")
    a_o = sp.MutableDenseNDimArray([a,0])
    o_b = sp.MutableDenseNDimArray([0,b])
    u_u = sp.MutableDenseNDimArray([1,1])
    u_o = sp.MutableDenseNDimArray([1,0])
    o_u = sp.MutableDenseNDimArray([0,1])
    a_b_diff = sp.MutableDenseNDimArray([a-b,b-a])
    a_b = sp.MutableDenseNDimArray([a,b])
    b_a = sp.MutableDenseNDimArray([b,a])
    
    from sympy import tensorproduct as tp
    from utils import *
    C_op = sp.MutableDenseNDimArray([[a,b],[b,a]])
    D_op = C_op
    C_op = forget(forget(blow(C_op),1),1)
    name = 'C_op'
    draw_general(C_op, name)