import sympy as sp
from utils import * 

alpha = sp.symbols("alpha")
beta = sp.symbols("beta")

B = sp.MutableDenseNDimArray([alpha,beta])
C = sp.MutableDenseNDimArray([[alpha,beta],[beta,alpha]])
E = C
A = sp.MutableDenseNDimArray([[[alpha,beta],[beta,alpha]],[[beta,alpha],[beta,alpha]]])
D = A

B = forget(forget(forget(blow(B),2),3),4)
C = forget(forget(blow(C),3),4)
D = forget(blow(D),4)
E = blow(forget(forget(E,0),1))

A = forget(forget(A,0),2)

T = BM_product([B,C,D,E,A])

print(T)

draw_general(T,'5_nodi')
