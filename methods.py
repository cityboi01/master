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
    A = spairMatToCoo(A)

    #Get the right-hand-side constraint coefficients
    b = np.array(lp_file.linear_constraints.get_rhs())
    
    
    return A,b,c

def getInstance(arr, index):
    c = np.array(arr[index*4])
    A = np.array(arr[index*4+1:index*4+3])
    b = np.array(arr[index*4+3])
    return A, b, c

#number of bits in representation per variable 
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
    diagC = np.diag(c)
    zeros = np.zeros(((kMax - 1) * n, n))
    adjDC = np.concatenate((diagC, zeros), axis=0)
    Qxx += (adjDC.dot(Tx))

    Qxs = p * np.transpose(Tx).dot(np.transpose(A)).dot(Ts)

    Qsx = np.transpose(Qxs)

    Qss = p * (np.transpose(Ts).dot(Ts) + np.diag(np.transpose(Ts).dot(b) + b.reshape(1, -1).dot(Ts)))

    Qx = np.concatenate((Qxx, Qxs), axis=1 )
    Qs = np.concatenate((Qsx, Qss), axis=1 )

    Q = np.concatenate((Qx, Qs), axis = 0)

    #converts the Q Qubo matrix into a dict of form (u,v): value as required by neal.SimulatedAnnealingSampler() docs.
    return dict(np.ndenumerate(Q))

#a cplex lp object has a constraint matrix variable where each row is of type cplex._internal_matrices.SparsePair (spair)
#this function takes such an matrix A as inpyt and transforms into a dict
def spairMatToDict(A):
    dict = {}
    for indRow, row in enumerate(A):
        tuple = row.unpack()
        for (col, val) in (zip(tuple[0], tuple[1])):
            dict[(indRow, col)] = val
    return dict

def dictToCoo(d):
    keys = d.keys()
    indices = list(zip(*keys))
    return coo_matrix((list(d.values()), (indices[0], indices[1])))

def spairMatToCoo(A):
    i=[]
    j=[]
    values=[]
    for  ind,row in enumerate(A):
        tuple = row.unpack()
        i.extend([ind] * len(tuple[0]))
        j.extend(tuple[0])
        values.extend(tuple[1])
    return coo_matrix((values, (i, j)))
    