"""
Authors : CADET Matthias, LEFEZ Alexis, HERMANN Marilyn
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
    """
   Cette fonction renvoie la matrice obtenue après l’application de
   la méthode de Gauss à A˜.

     Argument: Aaug: Matrice carrée

     Retourne: Une matrice augmentée de format (n, n + 1)
    """

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
    """
   Cette fonction renvoie la solution d’un système T X = B, où T est triangulaire supérieure. 

      Argument: Taug: Matrice augmentée de ce système de format (n, n + 1)

      Retourne: La solution du système TX = B
    """
        
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
    """
    Cette fonction renvoie la solution d’un système AX = B (B un vecteur colonne).

     Argument:  A: Une matrice carrée
                B: Un vecteur colonne

     Retourne: La solution du système AX = B

    """
        
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
    """
    Cette fonction renvoie la décomposition LU d’une matrice carrée A.

     Argument: A: Une matrice carrée

     Retourne: La décomposition LU de A

    """
        
    n, m = np.shape(A)
    L = np.eye(n, m)
    for i in range(0, n):
        for k in range(i + 1, n):
            g = A[k, i] / A[i, i]
            for j in range(0, n):
                A[k, j] = A[k, j] - g*A[i, j]
            L[k, i] = g

    return L, A


# A = np.array([[3,2,-1,4], [-3,-4,4,-2], [6,2,2,7], [9,4,2,18]])
# L, U = DecompostionLU(A)

"""
print(L)
print(A)
print(L.dot(U))
"""

# QUESTION 2

def ResolutionLU(L, U, B):
    """
    Cette fonction résoud AX = B avec la décomposition de A = LU fourni en argument.

    Argument: L: Une matrice triangulaire supérieure
           U: Une matrice triangulaire supérieure
           B: Un vecteur colonne
           
    Retourne: La solution de l'équation AX = B
    """
        
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

# B = np.array([4,-5,-2,13])

#print(ResolutionLU(L, U, B))



# QUESTION 4
TpsG = []
TpsL = []
TpsU = []
length = []
ErreurG = []
ErreurLU = []
ErreurLin = []

for n in range(100,1000,100):
    A = np.random.rand(n,n)
    B = np.random.rand(n,1)

    time_startG = time.time()
    X1 = Gauss(A, B)
    time_endG = time.time()
    TpsG.append(time_endG - time_startG)
    Y1 = np.linalg.norm(np.dot(A, X1) - B)
    ErreurG.append(Y1)

    time_startL = time.time()
    X2 = np.linalg.solve(A, B)
    time_endL = time.time()
    TpsL.append(time_endL - time_startL)
    Y2 = np.linalg.norm(np.dot(A, X2) - B)
    ErreurLin.append(Y2)

    time_startU = time.time()
    L, U = DecompostionLU(A)
    X3 = ResolutionLU(L, U, B)
    time_endU = time.time()
    TpsU.append(time_endU - time_startU)
    Y3 = np.linalg.norm(np.dot(A, X3) - B)
    ErreurLU.append(Y3)

    length.append(n)

plt.figure("Temps d'exécution en fonction de la taille de la matrice")
plt.plot(length, TpsG, label = "Gauss")
plt.plot(length, TpsL, label = "linalg")
plt.plot(length, TpsU, label = "LU")
plt.xlabel("Taille de la matrice(n)")
plt.ylabel("Temps(s)")
plt.title("Temps d'exécution en fonction de la taille de la matrice")
plt.legend()

plt.figure("Courbe d'erreurs")
plt.title("Erreur relative en fonction de la taille de la matrice")
plt.plot(length, ErreurG, label = "Gauss")
plt.plot(length, ErreurLin, label = "Linalg")
plt.plot(length, ErreurLU, label = "LU")
plt.xlabel("Taille de la matrice(n)")
plt.ylabel("Erreur relative")
plt.legend()
plt.show()


# -----------------------Variantes de l’algorithme de Gauss--------------------------

# QUESTION 1

def GaussChoixPivotPartiel(A, B):
    """
Cette fonction résoud l'équation AX = B avec le choix du pivot partiel. On utilise des échanges de lignes, pour que le pivot soit choisi de
plus grand module possible au sein de la colonne.


 Argument: A: Une matrice carrée
           B: Une matrice colonne
           
 Retourne: La solution sous la forme d'une matrice colonne
     """

    nA, mA = A.shape               
    nB, mB = B.shape

    if nA == mA:                   
        if mB == 1 and nB == nA:      
            Aaug = np.concatenate((A, B), axis=1)    
            n, m = Aaug.shape

            for j in range(0, m - 2):
                Max = j               

                for i in range(j, n):
                    if abs(Aaug[i, j]) > abs(Aaug[Max, j]):
                        Max = i

                temporaire = Aaug[j].copy()
                Aaug[j] = Aaug[Max]
                Aaug[Max] = temporaire

                for i in range(0, n):
                    p = Aaug[j, j]
                    
                    if i > j:
                        g = Aaug[i, j] / p
                        Aaug[i] = Aaug[i] - g * Aaug[j]
        else:
            print("B doit être une matrice colonne et avoir autant de lignes que A.")
            
    else:
        print("A doit être une matrice carré.")

    return ResolutionSystTriSup(Aaug)




def GaussChoixPivotTotal(A, B):
    """
Cette fonction rend la solution d’un système AX = B avec la méthode de Gauss avec choix de pivot total.

 Argument: A: Une matrice carrée
           B: Un vecteur colonne
           
 Retourne: La solution du système AX = B
 
    """
    nA, mA = A.shape
    nB, mB = B.shape

    if nA == mA:
        if mB == 1 and nB == nA:
            Aaug = np.concatenate((A, B), axis=1)
            n, m = Aaug.shape

            historique = [i for i in range(m - 1)]

            for j in range(0, m - 2):

                LePlusGrand = j, j 

                for j2 in range(j, m - 1):
                    for i2 in range(j, n):
                        if abs(Aaug[i2, j2]) > abs(Aaug[LePlusGrand]): 
                            LePlusGrand = i2, j2

 
                histo_tempo = historique[j]
                historique[j] = historique[LePlusGrand[1]]
                historique[LePlusGrand[1]] = histo_tempo

               
                temporaire = Aaug[j].copy()
                Aaug[j] = Aaug[LePlusGrand[0]]
                Aaug[LePlusGrand[0]] = temporaire

                temporaire = Aaug[:, j].copy()
                Aaug[:, j] = Aaug[:, LePlusGrand[1]]

                Aaug[:, LePlusGrand[1]] = temporaire
               

                for i in range(0, n):
                    p = Aaug[j, j]

                    if i > j:
                        g = Aaug[i, j] / p
                        Aaug[i] = Aaug[i] - g * Aaug[j]
        else:
            print("La matrice B doit être une matrice colonne et avoir autant de ligne que la matrice A")
    else:
        print("La matrice A doit être une matrice carré")


    X_desorsonne = ResolutionSystTriSup(Aaug)
    X_ordonne = [0] * (m - 1)
    for i in range(len(historique)): 
        X_ordonne[historique[i]] = X_desorsonne[i]

    return np.array(X_ordonne)


# création du graphique des temps d'exécution des méthodes
plt.plot(tailles_matrices, temps)  
    plt.title(fonction.__name__ + "\nTemps d'exécution en fonction de la taille de la matrice")
    plt.xlabel("Taille de la matrice")
    plt.ylabel("Temps d'exécution")
    plt.grid()
    plt.show()
 
    plt.semilogx(tailles_matrices, erreurs) 
