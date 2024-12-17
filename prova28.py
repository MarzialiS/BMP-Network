import sympy as sp
import numpy as np
from array import array
from itertools import product
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt



def forget(T, pos):
    ### This method takes in input an n-order tensor T and gives as output a n+1-order tensor T
    ### pos is an integer in the interval [0,lent(T.shape)] 
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

def general_forget(T, pos):
    N = T.shape[0] # assuming T is cubical 
    ### This method takes in input an n-order tensor T and gives as output a n+1-order tensor T
    ### pos is an integer in the interval [0,lent(T.shape)] 
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



def BM_product(tensor_list):
    # tensor_list must be ordered according to the order requested for the BM product
    # up to now it takes only tensors with indices in {0,1}
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

def general_BM_product(tensor_list):
    N = tensor_list[0].shape[0]
    # tensor_list must be ordered according to the order requested for the BM product
    # up to now it takes only tensors with indices in {0,1}
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
    return T_out

def flat(T):
    # trasforma i tensori [2,2,2,....] in [8,...]
    # utile per il plot dei tensori
    length = len(T.shape)
    if length<3:
        raise Exception('il tensore non è 3D')
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
    new_T[0] = T[[0,0,0]+list(index)]
    new_T[1] = T[[1,0,0]+list(index)]
    new_T[2] = T[[0,1,0]+list(index)]
    new_T[3] = T[[1,1,0]+list(index)]
    new_T[4] = T[[0,0,1]+list(index)]
    new_T[5] = T[[1,0,1]+list(index)]
    new_T[6] = T[[0,1,1]+list(index)]
    new_T[7] = T[[1,1,1]+list(index)]
    return new_T
    # 

def draw_tensor(T,ax):
    #per ora vale solo per quelli di ordine 3
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
    # vale soltanto per tensori fino a ordine 5
    length = len(T.shape)
    fig, axs = plt.subplots(2**(int(np.floor((length-3)/2))),2**(int(np.floor((length-2)/2))))
    print(axs)
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

# Elements of the JK matrix
a = sp.symbols("a")
b = sp.symbols("b")
c = sp.symbols("c")
d = sp.symbols("d")
g = sp.symbols("g")
h = sp.symbols("h")
t = sp.symbols("t")
al = sp.symbols("al")
be = sp.symbols("be")

# Unknown initial distribution
x = sp.symbols("x")
y = sp.symbols("y")
u = sp.symbols("u")
v = sp.symbols("v")


B = sp.MutableDenseNDimArray([a, c, g, t])
print("The initil distribution B is", B)

C = sp.MutableDenseNDimArray([[al, be, be, be], [be, al, be, be], [be, be, al, be], [be, be, be, al]])
print("The initil distribution C is", C)

A = sp.MutableDenseNDimArray([[al, be, be, be], [be, al, be, be], [be, be, al, be], [be, be, be, al]])
print("The initil distribution A is", A)

B1 = general_blow(B)
print("B1", B1)
B2 = general_forget(B1, 2)
print("B2", B2)

C1 = general_forget(C, 2)
print("C1", C1)

A1 = general_forget(A, 1)
print("A1", A1)
#print(size(A1))

T1 = general_BM_product([A1, B2, C1])
print("T1", T1)

CC1 = general_blow(C)
AA1 = general_forget(A, 0)

T2 = general_BM_product([AA1, B2, CC1])
print("T2", T2)

print(T1 - T2)

'''








AAA1 = sp.MutableDenseNDimArray([[[a,b],[c,c]], [[c,c],[b,a]]])
T2 = BM_product([AAA1, B2, CC1])
print("T2", T2)

BBBB1 = blow(B)
BBBB2 = forget(BBBB1, 2)
CCCC1 = forget(B, 0)
CCCC2 = blow(CCCC1)
T3 = BM_product([AAA1, BBBB2, CCCC2])
print("T3", T3)
'''