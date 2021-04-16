"""
Authors : CADET Matthias, LEFEZ Alexis, HERMANN Marilyn
Date : 16 avril
"""


# -----------------------------------
import numpy as np
import matplotlib.pyplot as plt
from math import *
from numba import njit
import time
# -----------------------------------

@njit
def ResolutionSystTriSup(Taug):
    """
   Cette fonction renvoie la solution d’un système T X = B, où T est triangulaire supérieure. 

      Argument: Taug: Matrice augmentée de ce système de format (n, n + 1)

      Retourne: La solution du système TX = B
    """

    n, m = np.shape(Taug)
    x = np.zeros(n)
    x[n - 1] = Taug[n - 1, m - 1] / Taug[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = Taug[i, m - 1]
        for j in range(i + 1, n):
            x[i] = x[i] - Taug[i, j] * x[j]
        x[i] = x[i] / Taug[i, i]
    return x


def ResolutionSystTriInferieur(Taug):
    n, m = np.shape(Taug)
    Y = np.zeros(n)

    for i in range(0, n):
        somme = 0
        for j in range(0, n):
            somme = somme + Y[j]*Taug[i, j]
        Y[i] = (Taug[i, n]-somme)/Taug[i, i]

    return Y


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
                Aaug[k, j] = Aaug[k, j] - g * Aaug[i, j]
            Aaug[k, n] = Aaug[k, n] - g * Aaug[i, n]
    return Aaug


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


def DecompositionLU(A):
    """
    Cette fonction renvoie la décomposition LU d’une matrice carrée A.

    Args:
        A : Une matrice carrée

    Returns:
        [Matrice]: décomposition LU de A
    """
    
    n, m = np.shape(A)
    L = np.eye(n)
    U = A

    for i in range(0, n-1):

        for j in range(i+1, n):
            pivot = U[j, i]/U[i, i]
            L[j, i] = pivot

            for k in range(i, n):
                U[j, k] = U[j, k] - pivot * U[i, k]

    return L, U


# QUESTION 2

def ResolutionLU(L, U, B):
    """
    Cette fonction résoud AX = B avec la décomposition de A = LU fourni en argument.

    Argument: L: Une matrice triangulaire supérieure
           U: Une matrice triangulaire supérieure
           B: Un vecteur colonne
           
    Retourne: La solution de l'équation AX = B
    """
    Aaug = np.concatenate((L, B), axis=1)
    n, m = np.shape(Aaug)
    Y = np.reshape(ResolutionSystTriInferieur(Aaug), (n, 1))
    Aaug = np.concatenate((U, Y), axis=1)
    X = ResolutionSystTriSup(Aaug)
    return X


def GaussChoixPivotPartiel(A, B):
    """
    Cette fonction résoud l'équation AX = B avec le choix du pivot partiel. On utilise des échanges de lignes, pour que le pivot soit choisi de
    plus grand module possible au sein de la colonne.


    Argument: A: Une matrice carrée
            B: Une matrice colonne

    Retourne: La solution sous la forme d'une matrice colonne
    """
    Aaug = np.concatenate((A, B), axis=1)
    n, m = np.shape(Aaug)

    for i in range(n):
        for j in range(0, n-i):
            if abs(Aaug[j, j]) < abs(Aaug[i+j, j]):
                i_max = Aaug[i, :].copy()
                Aaug[i, :] = Aaug[i+j, :]
                Aaug[i+j, :] = i_max
            for k in range(i+1, n):
                g = Aaug[k, i]/Aaug[i, i]
                Aaug[k, :] = Aaug[k, :] - g * Aaug[i, :]
    solution = ResolutionSystTriSup(Aaug)
    return solution


def ReductionGaussChoixPivotTotal(Aaug):
    n, m = np.shape(Aaug)

    for i in range(n):
        for j in range(0, n-i):
            if abs(Aaug[j, j]) < abs(Aaug[i+j, j]):
                i_max = Aaug[i, :].copy()
                Aaug[i, :] = Aaug[i+j, :]
                Aaug[i+j, :] = i_max
            if abs(Aaug[j, j]) < abs(Aaug[j, i+j]):
                J_max = Aaug[:, i].copy()
                Aaug[:, i] = Aaug[:, i+j]
                Aaug[:, i+j] = J_max
            for k in range(i+1, n):
                g = Aaug[k, i]/Aaug[i, i]
                Aaug[k, :] = Aaug[k, :] - g * Aaug[i, :]
    return Aaug


def GaussChoixPivotTotal(A, B):
    """
    Cette fonction rend la solution d’un système AX = B avec la méthode de Gauss avec choix de pivot total.

    Argument: A: Une matrice carrée
            B: Un vecteur colonne

    Retourne: La solution du système AX = B
    """
    Aaug = np.concatenate((A, B), axis=1)

    Taug = ReductionGaussChoixPivotTotal(Aaug)
    solution = ResolutionSystTriSup(Taug)

    return solution


def Cholesky(A):
    """[summary]

    Args:
        A ([type]): [description]

    Returns:
        [type]: [description]
    """
    n, m = np.shape(A)
    L = np.zeros((n, m))
    for k in range(0, n):
        for i in range(k, n):
            if k == i:
                S_diag = 0
                for j in range(0, k):
                    S_diag += L[k, j]**2
                L[k, k] = sqrt(A[k, k] - S_diag)
            else:
                S_non_diag = 0
                for j in range(0, k):
                    S_non_diag += L[i, j] * L[k, j]
                L[i, k] = (A[i, k] - S_non_diag) / L[k, k]

    return L


def ResolCholesky(A, B):
    """[summary]

    Args:
        A ([type]): [description]
        B ([type]): [description]

    Returns:
        [type]: [description]
    """
    L = Cholesky(A)
    L_T = np.transpose(L)

    n, m = np.shape(L)
    Y = np.zeros(n)
    X = np.zeros(n)
    for i in range(0, n):
        S = 0
        for k in range(0, i):
            S += L[i, k] * Y[k]
        Y[i] = (B[i] - S)/L[i, i]

    for j in range(n - 1, -1, -1):
        S = 0
        for k in range(j+1, n):
            S += L_T[j, k] * X[k]
        X[j] = (Y[j] - S) / L_T[j, j]
    return X


def graphes():
    """
    Cette fonction affiche les graphiques de temps d'execution en fonction de la taille
    de la matrice, des erreurs en fonction de la taille de la matrice et les graphiques
    loglog (temps d'execution - taille de la matrice)
    """
    TpsC = []
    TpsPT = []
    TpsLS = []
    TpsG = []
    TpsPP = []
    TpsLU = []
    ErreurC = []
    ErreurPP = []
    ErreurG = []
    ErreurPT = []
    ErreurLS = []
    ErreurLU = []
    length = []

    for n in range(100, 1000, 50):
        A = np.random.rand(n, n)
        B = np.random.rand(n, 1)

        M = np.dot(A, np.transpose(A))
        

        # CHOLESKY

        time_startC = time.time()
        X1 = ResolCholesky(M, B)
        time_endC = time.time()
        TpsC.append(time_endC - time_startC)
        Y1 = np.linalg.norm(np.dot(M, X1) - np.ravel(B))
        ErreurC.append(Y1)

        # PIVOT TOTAL

        time_startPT = time.time()
        X2 = GaussChoixPivotTotal(M, B)
        time_endPT = time.time()
        TpsPT.append(time_endPT - time_startPT)
        Y2 = np.linalg.norm(np.dot(M, X2) - np.ravel(B))
        ErreurPT.append(Y2)

        # LINALG.SOLVE

        time_startLS = time.time()
        X3 = np.linalg.solve(M, B)
        X3.reshape(n, 1)
        time_endLS = time.time()
        TpsLS.append(time_endLS - time_startLS)
        Y3 = np.linalg.norm(np.dot(M, np.ravel(X3)) - np.ravel(B))
        ErreurLS.append(Y3)

        # PIVOT PARTIEL

        time_startPP = time.time()
        X4 = GaussChoixPivotPartiel(M, B)
        time_endPP = time.time()
        TpsPP.append(time_endPP - time_startPP)
        Y4 = np.linalg.norm(np.dot(M, X4) - np.ravel(B))
        ErreurPP.append(Y4)

        # GAUSS

        time_startG = time.time()
        X5 = Gauss(M, B)
        time_endG = time.time()
        TpsG.append(time_endG - time_startG)
        Y5 = np.linalg.norm(np.dot(M, X5) - np.ravel(B))
        ErreurG.append(Y5)

        # LU

        time_startLU = time.time()
        L, U = DecompositionLU(M)
        X6 = ResolutionLU(L, U, B)
        time_endLU = time.time()
        TpsLU.append(time_endLU - time_startLU)
        Y6 = np.linalg.norm(np.dot(np.dot(L, U), X6) - np.ravel(B))
        ErreurLU.append(Y6)

        length.append(n)

    # PLOT

    plt.figure("Temps d'exécution en fonction de la taille de la matrice")
    plt.plot(length, TpsC, label="Cholesky")
    plt.plot(length, TpsPT, label="Pivot Total")
    plt.plot(length, TpsG, label="Gauss")
    plt.plot(length, TpsPP, label="Pivot Partiel")
    plt.plot(length, TpsLU, label="LU")
    plt.plot(length, TpsLS, label="Linalg.solve")
    plt.xlabel("Taille de la matrice(n)")
    plt.ylabel("Temps d'execution(s)")
    plt.title("Temps d'exécution en fonction de la taille de la matrice")
    plt.grid()
    plt.legend()

    # ERREURS

    plt.figure("Courbe d'erreurs")
    plt.title("Erreur relative en fonction de la taille de la matrice")
    plt.semilogy(length, ErreurC, label="Cholesky")
    plt.semilogy(length, ErreurPT, label="Pivot Total")
    plt.semilogy(length, ErreurG, label="Gauss")
    plt.semilogy(length, ErreurPP, label="Pivot Partiel")
    plt.semilogy(length, ErreurLU, label="LU")
    plt.semilogy(length, ErreurLS, label="Linalg.solve")
    plt.xlabel("Taille de la matrice(n)")
    plt.ylabel("Erreur relative")
    plt.grid()
    plt.legend()

    # GRAPHES LOLGLOG

    plt.figure("Graphes loglog")
    plt.loglog(length, TpsC, label="Cholesky")
    plt.loglog(length, TpsPT, label="Pivot Total")
    plt.loglog(length, TpsG, label="Gauss")
    plt.loglog(length, TpsPP, label="Pivot Partiel")
    plt.loglog(length, TpsLU, label="LU")
    plt.loglog(length, TpsLS, label="Linalg.solve")
    plt.title("Temps d'exécution en fonction de la taille de la matrice (log-log)")
    plt.xlabel("Taille de la matrice(n)")
    plt.ylabel("Temps d'exécution(s)")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    graphes()
