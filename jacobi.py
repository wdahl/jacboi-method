from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot, linalg 
import math

def jacobi(A,b,N, tol, sol, x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed                                                                                                                                                            
    if x is None:
        x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = diag(A)
    R = A - diagflat(D)

    # Iterate for N times                                                                                                                                                                          
    for i in range(N):
        error = max(sol-x)
        print(f'k: {i}')
        print(f'x: {x}')
        print(f'Error: {error}')
        if error < tol:
            return x

        x = (b - dot(R,x)) / D
    return x

A = array([[4,1, 1, 0, 0],[-1, -3, 1, 1, 0],[2, 1, 5, -1, -1],[-1, -1, -1, 4, 0],[0, 2, -1, 1, 4]])
b = array([6, 6, -6, 6, 6])
TOL = math.pow(10, -3)
a_sol = [2.4063745019920315, -2.7091633466135456, -0.9163346613545813, 1.195219123505976, 2.326693227091633]

sol = jacobi(A,b,25,TOL, a_sol)

print("A:")
pprint(A)

print("b:")
pprint(b)

print("x:")
pprint(sol)