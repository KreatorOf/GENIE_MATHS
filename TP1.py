"""
Authors : CADET Matthias, FOLY Harold
Date : 28/01/2021
Professor : MR. BLETZACKER
"""

#-----------------------------------------------------------
#                        BIBLIOTHEQUES
#-----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import time

#-----------------------------------------------------------
#                          FONCTIONS
#-----------------------------------------------------------

""" def echanger(A, L1, L2): #échange les lignes L1 et L2 de la matrice
    temp = A[L1]
    A[L1] = A[L2]
    A[L2] = temp

def zero(A, L1, L2, k):
    for i in range(len(A) + 1): # len(A) + 1 car la matrice est augmentée
        A[L2][]
"""
# QUESTION 1

def ReductionGauss(Aaug):
    n, m = np.shape(Aaug)
    for i in range(0, n):
        for k in range(i + 1, n):
            g = Aaug[k, i] / Aaug[i, i]
            for j in range(0, n):
                Aaug[k, j] = Aaug[k, j] - g*Aaug[i, j]
            Aaug[k, n] = Aaug[k, n] - g*Aaug[i, n]  
    return Aaug



# QUESTION 2
 
def ResolutionSystTriSup(Taug):
    n, m = np.shape(Taug)
    x = np.zeros(n)
    x[n - 1] = Taug[n - 1, m - 1]/Taug[n - 1, n - 1]
    for i in range (n - 2, -1, -1):
        x[i] = Taug[i, m - 1]
        for j in range (i + 1, n): 
            x[i] = x[i] - Taug[i, j]*x[j]
        x[i] = x[i]/Taug[i, i]
    return x



# QUESTION 3

def Gauss(A, B):
    n, m = np.shape(A)
    R = np.c_[A, B]
    return ResolutionSystTriSup(ReductionGauss(R))
"""
A = np.array([[2,5,6], [4,11,9], [-2,-8,7]])
B = np.array([7,12,3])
print(Gauss(A, B))
"""




# -----------------------DECOMPOSITION LU--------------------------

# QUESTION 1

def DecompostionLU(A):
    n, m = np.shape(A)
    L = np.eye(n, m)
    for i in range(0, n):
        for k in range(i + 1, n):
            g = A[k, i] / A[i, i]
            for j in range(0, n):
                A[k, j] = A[k, j] - g*A[i, j]
            L[k, i] = g

    return L, A


A = np.array([[3,2,-1,4], [-3,-4,4,-2], [6,2,2,7], [9,4,2,18]])
L, U = DecompostionLU(A)

"""
print(L)
print(A)
print(L.dot(U))
"""

# QUESTION 2

def ResolutionLU(L, U, B):
    n, m = np.shape(L)
    Y = np.zeros(n)
    X = np.zeros(n)
    for i in range(0, n):
        S = 0
        for k in range(0, i):
            S += L[i, k]*Y[k]
        Y[i] = B[i] - S

    for j in range(n - 1, -1, -1):
        S=0
        for k in range(j, n):
            S += U[j, k]*X[k]
        X[j] = (Y[j] - S) / U[j,j]
    return X

B = np.array([4,-5,-2,13])

#print(ResolutionLU(L, U, B))



# QUESTION 4

TpsG = []
TpsL = []
TpsU = []
length = []

for n in range(2,300,5):
    A = np.random.rand(n,n)
    B = np.random.rand(n,1)

    time_startG = time.time()
    Gauss(A, B)
    time_endG = time.time()
    TpsG.append(time_endG - time_startG)

    time_startL = time.time()
    np.linalg.solve(A, B)
    time_endL = time.time()
    TpsL.append(time_endL - time_startL)

    time_startU = time.time()
    L, U = DecompostionLU(A)
    ResolutionLU(L,U,B)
    time_endU = time.time()
    TpsU.append(time_endU - time_startU)

    length.append(n)

plt.plot(length, TpsG, label = "Gauss")
plt.plot(length, TpsL, label = "linalg")
plt.plot(length, TpsU, label = "LU")
plt.xlabel("n")
plt.ylabel("Temps(s)")
plt.title("Temps d'éxecution en fonction de la taille de la matrice")
plt.legend()
plt.show()
#print(A)
#print(B)