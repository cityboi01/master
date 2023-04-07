import numpy as np
from scipy.sparse import coo_matrix

def importLP(file):
    import cplex

    #Read the cplex .lp file
    lp_file = cplex.Cplex(file)

    #Get the objective coefficients
    c = np.array(lp_file.objective.get_linear())

    #Get the left-hand-side constraint coefficient matrix
    A = np.array(lp_file.linear_constraints.get_rows())

    #Get the right-hand-side constraint coefficients
    b = np.array(lp_file.linear_constraints.get_rhs())
    
    
    return A,b,c

def getInstance(arr, index):
    c = np.array(arr[index*4])
    A = np.array(arr[index*4+1:index*4+3])
    b = np.array(arr[index*4+3])
    return A, b, c

#number of bits need in representation per variable  if i is max value
def R (i):
    k=1
    while(pow(2,k) < i):
        k+=1
    return k

#kMax=1 (1 bit (0,1): returns identiy (n variables are binary
#n is numVariables in x
#m is numVariables in s (numConstraints)
#kMax is number of bits per variables (it defines varibales maxValue by powe(2,kMax) - 1))
def bitMap(n, m, kMax):
    T = np.zeros((n+m, (n+m)*kMax))
    print(type(T))
    indent = 0
    for index,x in enumerate(T):
        for y in range(0,kMax):
            T[index, y + indent*kMax] = pow(2,y)
        indent += 1
    return T 

def Qdict(A,b,c,kMax,p):
    n = c.size
    m = b.size

    #computes bit mappings of variables x and slacks s
    T = bitMap(n, m, kMax)
    Tx = T[0:n,0:n*kMax]
    Ts = T[n:T.size, n*kMax:T[0].size]
        

    #Computes the block matrices in Q
    Qxx = (p * np.transpose(Tx).dot(np.transpose(A).dot(A) + np.diag(np.transpose(A).dot(b) + b.reshape(1,-1).dot(A)))).dot(Tx)
    diag = np.diag(np.transpose(c).dot(Tx))
    Qxx += diag

    Qxs = p * np.transpose(Tx).dot(np.transpose(A)).dot(Ts)

    Qsx = np.transpose(Qxs)

    Qss = p * (np.transpose(Ts).dot(Ts) + np.diag(np.transpose(Ts).dot(b) + b.reshape(1, -1).dot(Ts)))

    Qx = np.concatenate((Qxx, Qxs), axis=1 )
    Qs = np.concatenate((Qsx, Qss), axis=1 )

    Q = np.concatenate((Qx, Qs), axis = 0)

    #converts the Q Qubo matrix into a dict of form (u,v): value as required by neal.SimulatedAnnealingSampler() docs.
    return dict(np.ndenumerate(Q))

