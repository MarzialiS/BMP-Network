import sympy as sp
import numpy as np


# Initial distribution of the first node A
p1 = sp.symbols("p1")
q1 = sp.symbols("q1")
pA = sp.symbols("pA")
qA = sp.symbols("qA")

# Initial distribution of the second node B
p2 = sp.symbols("p2")
q2 = sp.symbols("q2")
pB = sp.symbols("pB")
qB = sp.symbols("qB")

# Initial distribution of the second node C
pC = sp.symbols("pC")
qC = sp.symbols("qC")

# Initial distribution of the second node D
pD = sp.symbols("pD")
qD = sp.symbols("qD")

# Initial distribution of the second node E
pE = sp.symbols("pE")
qE = sp.symbols("qE")

# Elements of the Mendel tensor
a = sp.symbols("a")
b = sp.symbols("b")
c = sp.symbols("c")
d = sp.symbols("d")

# In case the tensor is also Jukes-Cantor
# c=d

# Elements of the perturbation matrix Gamma between B and A or C and A
g = sp.symbols("g")
g1 = sp.symbols("g1")
h = sp.symbols("h")
l = sp.symbols("l")
m = sp.symbols("m")
n = sp.symbols("n")

# Decomment in case no perturbation matrix between nodes is detected
# g = 1
# g1 = 0

# Mendel activation tensor 
A = sp.MutableDenseNDimArray([[[[[pA, qA], [pA, qA]], [[pA, qA], [pA, qA]]], [[[pA, qA], [pA, qA]], [[pA, qA], [pA, qA]]]], 
                             [[[[pA, qA], [pA, qA]], [[pA, qA], [pA, qA]]], [[[pA, qA], [pA, qA]], [[pA, qA], [pA, qA]]]]])
print("The tensor A is", A)

# Tensor B
B = sp.MutableDenseNDimArray([[[[[pB, pB], [pB, pB]], [[pB, pB], [pB, pB]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]], 
                             [[[[qB, qB], [qB, qB]], [[qB, qB], [qB, qB]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]]])
print("The tensor B is", B)

# Tensor C
# C = sp.MutableDenseNDimArray([[[[[pC, pC], [pC, pC]], [[0, 0], [0, 0]]], [[[qC, qC], [qC, qC]], [[0, 0], [0, 0]]]], 
#                              [[[[0, 0], [0, 0]], [[pC, pC], [pC, pC]]], [[[0, 0], [0, 0]], [[qC, qC], [qC, qC]]]]])
C = sp.MutableDenseNDimArray([[[[[pC, pC], [pC, pC]], [[0, 0], [0, 0]]], [[[qC, qC], [qC, qC]], [[0, 0], [0, 0]]]], 
                             [[[[0, 0], [0, 0]], [[pC, pC], [pC, pC]]], [[[0, 0], [0, 0]], [[qC, qC], [qC, qC]]]]])
#print("The tensor C is", C)

# Tensor D
#D = sp.MutableDenseNDimArray([[[[[[pD, pD], [0, 0]], [[qD, qD], [0, 0]]], [[[pD, pD], [0, 0]], [[qD, qD], [0, 0]]]], 
#                              [[[[0, 0], [pD, pD]], [[0, 0], [qD, qD]]], [[[0, 0], [pD, pD]], [[0, 0], [qD, qD]]]]]])
D = sp.MutableDenseNDimArray([[[[[pD, pD], [0, 0]], [[qD, qD], [0, 0]]], [[[pD, pD], [0, 0]], [[qD, qD], [0, 0]]]], 
                             [[[[0, 0], [pD, pD]], [[0, 0], [qD, qD]]], [[[0, 0], [pD, pD]], [[0, 0], [qD, qD]]]]])
#print("The tensor D is", D)
print(D.shape)

# Tensor E
# E = sp.MutableDenseNDimArray([[[[[pE, 0], [qE, 0]], [[pE, 0], [qE, 0]]], [[[pE, 0], [qE, 0]], [[pE, qE], [0, 0]]]], 
#                              [[[[0, pE], [0, qE]], [[0, pE], [0, qE]]], [[[0, pE], [0, qE]], [[0, pE], [0, qE]]]]])
E = sp.MutableDenseNDimArray([[[[[pE, 0], [qE, 0]], [[pE, 0], [qE, 0]]], [[[pE, 0], [qE, 0]], [[pE, qE], [0, 0]]]], 
                             [[[[0, pE], [0, qE]], [[0, pE], [0, qE]]], [[[0, pE], [0, qE]], [[0, pE], [0, qE]]]]])
#print("The tensor E is", E)
print(E.shape)

# V = sp.MutableDenseNDimArray([[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]], 
#                               [[[17, 18], [19, 20]], [[21, 22], [23, 24]], [[25, 26], [27, 28]], [[29, 30], [31, 32 ]]]])
# print("The tensor V is", V)

# Searched tensor T
T = sp.MutableDenseNDimArray(np.zeros((2,2,2,2,2)))

V = sp.MutableDenseNDimArray([[[[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]])
# Triple tensor product
print(T.shape)
for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        for k in range(T.shape[2]):
            for l in range(T.shape[2]):
                for m in range(T.shape[2]):
                    for u in range(T.shape[0]):
                        #print(i, j, k, l, m, u)
                        T[i, j, k, l, m] += A[u, j, k, l, m]*B[i, u, k, l, m]*C[i, j, u, l, m]*D[i, j, k, u, m]*E[i, j, k, l, u]
print("The tensor T is", T)



