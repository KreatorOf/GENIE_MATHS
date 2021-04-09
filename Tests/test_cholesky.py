import sys
sys.path.append(".")

from Src import Cholesky
import numpy as np   
import unittest

class TestFunctionOption(unittest.TestCase):
    
    def test_cholesky(self):
        A = np.array([[1,2,4], [2,8,4], [4,4,24]])
        L = Cholesky.Cholesky(A)
        L_machine = np.linalg.cholesky(A)
        try:
            (L==L_machine).all()
        except:
            print("Les matrices ne sont pas égales")
            
    def test_ResolCholesky(self):
        A = np.array([[1,2,4], [2,8,4], [4,4,24]])
        B = np.array([1,2,3])
        X = Cholesky.ResolCholesky(A, B)
        X_verif = np.array([[ 2.5  -0.25 -0.25]])
        try:
            (X==X_verif).all()
        except:
            print("Les matrices ne sont pas égales")
    
    def test_cholesky_random3x3(self):
        A = np.random.rand(3,3)
        B = np.random.rand(3,1)
        
        S = np.dot(A, np.transpose(A)) #Matrice symétrique définie positive
        
        L = Cholesky.Cholesky(S)
        try:
            (np.dot(L, np.transpose(L))==S).all()
        except:
            print("La factorisation de Cholesky n'a pas fonctionné")
        
        
        
        
        

if __name__ == '__main__':
    unittest.main()