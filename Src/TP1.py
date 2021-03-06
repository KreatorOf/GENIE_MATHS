"""
Authors : CADET Matthias, LEFEZ Alexis, HERMANN Marilyn
Date : 28/01/2021
Professor : MR. BLETZACKER
"""

# -----------------------------------------------------------
#                        BIBLIOTHEQUES
# -----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import time
# -----------------------------------------------------------
#                          FONCTIONS
# -----------------------------------------------------------

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
                Aaug[k, j] = Aaug[k, j] - g * Aaug[i, j]
            Aaug[k, n] = Aaug[k, n] - g * Aaug[i, n]
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
    x[n - 1] = Taug[n - 1, m - 1] / Taug[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = Taug[i, m - 1]
        for j in range(i + 1, n):
            x[i] = x[i] - Taug[i, j] * x[j]
        x[i] = x[i] / Taug[i, i]
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
                A[k, j] = A[k, j] - g * A[i, j]
            L[k, i] = g

    return L, A


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
            S += L[i, k] * Y[k]
        Y[i] = B[i] - S

    for j in range(n - 1, -1, -1):
        S = 0
        for k in range(j, n):
            S += U[j, k] * X[k]
        X[j] = (Y[j] - S) / U[j, j]
    return X


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

    X_desordonne = ResolutionSystTriSup(Aaug)
    X_ordonne = [0] * (m - 1)
    for i in range(len(historique)):
        X_ordonne[historique[i]] = X_desordonne[i]

    return np.array(X_ordonne)


# QUESTION 4
# AFFICHAGE DES GRAPHES

TpsG = []
TpsLin = []
TpsLU = []
TpsPP = []
TpsPT = []

length = []

ErreurG = []
ErreurLin = []
ErreurLU = []
ErreurPP = []
ErreurPT = []

for n in range(100, 1000, 100):
    A = np.random.rand(n, n)
    B = np.random.rand(n, 1)


    # Gauss

    time_startG = time.time()
    X1 = Gauss(A, B)
    time_endG = time.time()
    TpsG.append(time_endG - time_startG)
    Y1 = np.linalg.norm(np.dot(A, X1) - B)
    ErreurG.append(Y1)

    # Linalg

    time_startLin = time.time()
    X2 = np.linalg.solve(A, B)
    time_endLin = time.time()
    TpsLin.append(time_endLin - time_startLin)
    Y2 = np.linalg.norm(np.dot(A, X2) - B)
    ErreurLin.append(Y2)

    # LU

    time_startLU = time.time()
    L, U = DecompostionLU(A)
    X3 = ResolutionLU(L, U, B)
    time_endLU = time.time()
    TpsLU.append(time_endLU - time_startLU)
    Y3 = np.linalg.norm(np.dot(A, X3) - B)
    ErreurLU.append(Y3)

    # Pivot Partiel

    time_startPP = time.time()
    X4 = GaussChoixPivotPartiel(A, B)
    time_endPP = time.time()
    TpsPP.append(time_endPP - time_startPP)
    Y4 = np.linalg.norm(np.dot(A, X4) - B)
    ErreurPP.append(Y4)

    # Pivot Total

    time_startPT = time.time()
    X5 = GaussChoixPivotTotal(A, B)
    time_endPT = time.time()
    TpsPT.append(time_endPT - time_startPT)
    Y5 = np.linalg.norm(np.dot(A, X5) - B)
    ErreurPT.append(Y5)

    # Taille de la matrice

    length.append(n)

# PLOT

plt.figure("Temps d'exécution en fonction de la taille de la matrice")
plt.plot(length, TpsG, label="Gauss")
plt.plot(length, TpsLin, label="Linalg")
plt.plot(length, TpsLU, label="LU")
plt.plot(length, TpsPP, label="Pivot Partiel")
plt.plot(length, TpsPT, label="Pivot Total")
plt.xlabel("Taille de la matrice(n)")
plt.ylabel("Temps d'execution(s)")
plt.title("Temps d'exécution en fonction de la taille de la matrice")
plt.grid()
plt.legend()

# ERREURS

plt.figure("Courbe d'erreurs")
plt.title("Erreur relative en fonction de la taille de la matrice")
plt.plot(length, ErreurG, label="Gauss")
plt.plot(length, ErreurLU, label="LU")
plt.plot(length, ErreurPP, label="Pivot Partiel")
plt.plot(length, ErreurPT, label="Pivot Total")
plt.xlabel("Taille de la matrice(n)")
plt.ylabel("Erreur relative")
plt.grid()
plt.legend()

# GRAPHES LOLGLOG

plt.figure("Graphes loglog")
plt.loglog(length, TpsG, label="Gauss")
plt.loglog(length, TpsLU, label="LU")
plt.loglog(length, TpsPP, label="Pivot Partiel")
plt.loglog(length, TpsPT, label="Pivot Total")
plt.title("Temps d'exécution en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice(n)")
plt.ylabel("Temps d'exécution(s)")
plt.grid()
plt.legend()
plt.show()
