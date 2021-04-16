from Src import TP2
import numpy as np
import numpy.testing as npt   
import unittest

class TestFunctionOption(unittest.TestCase):
    
    def test_Cholesky(self):
        A = np.array([[1,2,4], [2,8,4], [4,4,24]])
        L = TP2.Cholesky(A)
        L_machine = np.linalg.cholesky(A)
        npt.assert_array_equal(L, L_machine)
        
            
    def test_ResolCholesky(self):
        A = np.array([[1,2,4], [2,8,4], [4,4,24]])
        B = np.array([1,2,3])
        X = TP2.ResolCholesky(A, B)
        X_verif = np.array([2.5, -0.25, -0.25])
        npt.assert_array_equal(X, X_verif)

    
    def test_Cholesky_random3x3(self):
        A = np.random.rand(3,3)        
        S = np.dot(A, np.transpose(A)) #Matrice symétrique définie positive
        
        L = TP2.Cholesky(S)
        P = np.dot(L, np.transpose(L))
        npt.assert_array_almost_equal(P, S)

        
        
if __name__ == '__main__':
    unittest.main()
